# precompute_topic_shift_features.py
"""
Compute topic shift probabilities for QEvasion dataset using trained TIAGE classifier.

This script:
1. Loads the dataset with NLI features (data/qevasion_with_nli)
2. Uses the trained topic shift classifier to compute p_topic_shift
3. Saves the enhanced dataset to data/qevasion_with_nli_topic

Usage:
    python precompute_topic_shift_features.py
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def compute_topic_shift_features(
    topic_model_path: str = "models/topic_shift_roberta",
    in_dataset: str = "data/qevasion_with_nli",
    out_dataset: str = "data/qevasion_with_nli_topic",
    batch_size: int = 32,
    max_length: int = 256,
):
    """
    Compute topic shift probabilities for QEvasion dataset.
    """

    print("=" * 70)
    print("Computing Topic Shift Features for QEvasion")
    print("=" * 70)

    # Check if topic shift model exists
    topic_model_path = Path(topic_model_path)
    if not topic_model_path.exists():
        raise FileNotFoundError(
            f"Topic shift model not found at {topic_model_path}\n"
            f"Please train it first:\n"
            f"  python train_topic_shift_tiage.py --use_mock_data"
        )

    # Check if input dataset exists
    in_dataset_path = Path(in_dataset)
    if not in_dataset_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found at {in_dataset_path}\n"
            f"Please run precompute_nli_features.py first:\n"
            f"  python precompute_nli_features.py"
        )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load topic shift model
    print(f"\nLoading topic shift model from {topic_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(topic_model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(topic_model_path)).to(device)
    model.eval()
    print("  Model loaded successfully")

    # Load QEvasion dataset with NLI features
    print(f"\nLoading QEvasion dataset from {in_dataset_path}...")
    dataset = load_from_disk(str(in_dataset_path))
    print(f"  Loaded splits: {list(dataset.keys())}")
    for split in dataset.keys():
        print(f"    {split}: {len(dataset[split])} examples")

    # Define mapping function
    def add_topic_shift_prob(batch):
        """
        Compute p_topic_shift for each example.

        Context design: Use question as "context" and answer as "current"
        This makes sense for political QA - we check if answer shifts topic from question.
        """
        # Tokenize (context=question, current=answer)
        enc = tokenizer(
            batch["interview_question"],  # context
            batch["interview_answer"],    # current utterance
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()

        # Extract probability of "shift" (label=1)
        p_shift = probs[:, 1].tolist()

        return {"p_topic_shift": p_shift}

    # Compute topic shift features for each split
    new_splits = {}
    for split in dataset.keys():
        print(f"\nComputing topic shift features for split '{split}'...")
        print(f"  {len(dataset[split])} examples")

        ds_with_ts = dataset[split].map(
            add_topic_shift_prob,
            batched=True,
            batch_size=batch_size,
        )

        new_splits[split] = ds_with_ts

        # Show some statistics
        p_shifts = ds_with_ts["p_topic_shift"]
        print(f"  Topic shift probability stats:")
        print(f"    Mean: {sum(p_shifts)/len(p_shifts):.3f}")
        print(f"    Min:  {min(p_shifts):.3f}")
        print(f"    Max:  {max(p_shifts):.3f}")

    # Create new dataset with topic shift features
    new_dataset = DatasetDict(new_splits)

    # Save to disk
    out_dataset_path = Path(out_dataset)
    out_dataset_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving enhanced dataset to {out_dataset_path}...")
    new_dataset.save_to_disk(str(out_dataset_path))

    print("\n" + "=" * 70)
    print("Topic shift features computed successfully!")
    print("=" * 70)
    print(f"\nDataset saved to: {out_dataset_path}")
    print("\nFeatures in dataset:")
    print("  - p_contra (NLI: contradiction)")
    print("  - p_neutral (NLI: neutral)")
    print("  - p_entail (NLI: entailment)")
    print("  - p_topic_shift (NEW: topic shift probability)")
    print("\nNext step:")
    print("  python base_nli_topic.py")

    # Show a few examples
    print("\n" + "=" * 70)
    print("Sample examples:")
    print("=" * 70)
    train_sample = new_dataset["train"].select(range(min(3, len(new_dataset["train"]))))
    for i, ex in enumerate(train_sample):
        print(f"\nExample {i+1}:")
        print(f"  Question: {ex['interview_question'][:100]}...")
        print(f"  Answer:   {ex['interview_answer'][:100]}...")
        print(f"  p_topic_shift: {ex['p_topic_shift']:.3f}")
        print(f"  p_contra:      {ex['p_contra']:.3f}")
        print(f"  p_neutral:     {ex['p_neutral']:.3f}")
        print(f"  p_entail:      {ex['p_entail']:.3f}")
        print(f"  Clarity label: {ex['clarity_label']}")


def main():
    parser = argparse.ArgumentParser(description="Compute topic shift features for QEvasion")
    parser.add_argument("--topic_model_path", type=str, default="models/topic_shift_roberta",
                        help="Path to trained topic shift classifier")
    parser.add_argument("--in_dataset", type=str, default="data/qevasion_with_nli",
                        help="Input dataset path (with NLI features)")
    parser.add_argument("--out_dataset", type=str, default="data/qevasion_with_nli_topic",
                        help="Output dataset path (with NLI + topic shift features)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")

    args = parser.parse_args()

    compute_topic_shift_features(
        topic_model_path=args.topic_model_path,
        in_dataset=args.in_dataset,
        out_dataset=args.out_dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
