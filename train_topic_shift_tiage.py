# train_topic_shift_tiage.py
"""
Train a topic shift binary classifier on TIAGE dataset.

This classifier will be used to compute p_topic_shift feature for QEvasion dataset.

Usage:
    python train_topic_shift_tiage.py --tiage_root ../tiage
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def load_tiage_data(tiage_root: str) -> Dict[str, List[Dict]]:
    """
    Load TIAGE dataset and convert to (context, current, label) format.

    Expected TIAGE structure:
        tiage/
            data/
                train.json  (or similar format)
                dev.json
                test.json

    Each file contains dialogs with utterances and shift annotations.
    """
    tiage_path = Path(tiage_root)

    # Use PersonaChat annotated data
    train_file = tiage_path / "data" / "personachat" / "anno" / "train" / "anno_train.json"
    dev_file = tiage_path / "data" / "personachat" / "anno" / "dev" / "anno_dev.json"

    if not train_file.exists():
        raise FileNotFoundError(
            f"TIAGE train data not found at {train_file}\n"
            f"Please clone TIAGE repo first:\n"
            f"  git clone https://github.com/HuiyuanXie/tiage.git"
        )

    print(f"Loading TIAGE PersonaChat data from:")
    print(f"  Train: {train_file}")
    print(f"  Dev: {dev_file if dev_file.exists() else 'Not found, will split from train'}")

    def parse_tiage_json(file_path):
        """Parse TIAGE PersonaChat JSON format: {dialog_id: [[utt, label], ...]}"""
        examples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for dialog_id, turns in data.items():
            # turns is a list of [utterance, label]
            for i in range(1, len(turns)):
                # Skip first turn (label = -1)
                if turns[i][1] == "-1":
                    continue

                # Context: previous utterance(s)
                context_start = max(0, i - 1)  # Use last 1 utterance as context
                context = " ".join([t[0] for t in turns[context_start:i]])
                current = turns[i][0]

                # Label: "0" = no shift, "1" = shift
                label = 1 if turns[i][1] == "1" else 0

                examples.append({
                    "context": context,
                    "current": current,
                    "label": label,
                })

        return examples

    # Load train and dev
    splits = {}
    splits["train"] = parse_tiage_json(train_file)

    if dev_file.exists():
        splits["dev"] = parse_tiage_json(dev_file)
    else:
        # Split train into train/dev (90/10)
        all_train = splits["train"]
        split_idx = int(len(all_train) * 0.9)
        splits["train"] = all_train[:split_idx]
        splits["dev"] = all_train[split_idx:]

    print(f"Loaded {len(splits['train'])} training examples")
    print(f"Loaded {len(splits['dev'])} dev examples")

    # Print label distribution
    train_labels = [ex["label"] for ex in splits["train"]]
    print(f"Train label distribution: {sum(train_labels)} shifts / {len(train_labels)} total "
          f"({100*sum(train_labels)/len(train_labels):.1f}%)")

    # Store labels in splits for class weight computation
    splits["train_labels"] = train_labels

    return splits


def create_mock_tiage_data() -> Dict[str, List[Dict]]:
    """
    Create mock TIAGE-like data for testing when actual TIAGE is not available.
    This simulates topic shift detection task.
    """
    print("[WARNING] Using MOCK TIAGE data for testing purposes.")
    print("For real training, please download TIAGE dataset.")

    # Mock examples with clear topic shifts
    examples = [
        # No shift examples
        {"context": "What is your opinion on healthcare?",
         "current": "I believe healthcare should be accessible to everyone.", "label": 0},
        {"context": "Tell me about your economic policy.",
         "current": "Our economic policy focuses on job creation and growth.", "label": 0},
        {"context": "Do you support this bill?",
         "current": "Yes, I support this legislation because it helps families.", "label": 0},

        # Shift examples
        {"context": "What is your stance on immigration?",
         "current": "Let me tell you about our jobs program instead.", "label": 1},
        {"context": "Will you raise taxes?",
         "current": "The real question is what my opponent did last year.", "label": 1},
        {"context": "Do you support gun control?",
         "current": "I think we should focus on mental health.", "label": 1},
    ] * 100  # Repeat to have enough data

    # Shuffle
    import random
    random.shuffle(examples)

    split_idx = int(len(examples) * 0.8)
    return {
        "train": examples[:split_idx],
        "dev": examples[split_idx:],
    }


def compute_metrics(eval_pred):
    """Compute F1, precision, recall for binary classification."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="binary", pos_label=1),
        "precision": precision_score(labels, predictions, average="binary", pos_label=1, zero_division=0),
        "recall": recall_score(labels, predictions, average="binary", pos_label=1, zero_division=0),
    }


class WeightedLossTrainer(Trainer):
    """Custom Trainer with class-weighted loss for handling imbalanced data."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Use weighted cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def train_topic_shift_classifier(
    tiage_root: str,
    model_name: str = "roberta-base",
    output_dir: str = "models/topic_shift_roberta",
    num_epochs: int = 3,
    train_batch_size: int = 32,
    eval_batch_size: int = 64,
    learning_rate: float = 5e-5,
    max_length: int = 256,
    use_mock_data: bool = False,
):
    """Train topic shift classifier on TIAGE dataset."""

    print("=" * 70)
    print("Training Topic Shift Classifier on TIAGE")
    print("=" * 70)

    # Load data
    if use_mock_data or not Path(tiage_root).exists():
        print(f"\nTIAGE root '{tiage_root}' not found or using mock data.")
        raw_data = create_mock_tiage_data()
        # Extract labels for mock data
        train_labels = [ex["label"] for ex in raw_data["train"]]
    else:
        raw_data = load_tiage_data(tiage_root)
        train_labels = raw_data["train_labels"]

    # Compute class weights to handle imbalance
    unique_labels = np.array([0, 1])
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    print(f"\nClass imbalance detected:")
    print(f"  Class 0 (no_shift): {len([l for l in train_labels if l == 0])} samples, weight: {class_weights[0]:.3f}")
    print(f"  Class 1 (shift):    {len([l for l in train_labels if l == 1])} samples, weight: {class_weights[1]:.3f}")

    # Convert to HF Dataset
    dataset = DatasetDict({
        "train": Dataset.from_list(raw_data["train"]),
        "validation": Dataset.from_list(raw_data["dev"]),
    })

    # Tokenize
    print(f"\nTokenizing with {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess(batch):
        return tokenizer(
            batch["context"],
            batch["current"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    encoded = dataset.map(preprocess, batched=True)
    encoded = encoded.rename_column("label", "labels")
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Load model
    print(f"\nLoading {model_name} for sequence classification...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "no_shift", 1: "shift"},
        label2id={"no_shift": 0, "shift": 1},
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",  # Changed from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        seed=42,
    )

    # Determine device for class weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = class_weights_tensor.to(device)

    # Trainer with class weighting
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights_tensor,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Final evaluation
    print("\nFinal evaluation on validation set:")
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    print(f"\nModel saved to: {output_dir}")
    print("\nNext step:")
    print("  python precompute_topic_shift_features.py")


def main():
    parser = argparse.ArgumentParser(description="Train topic shift classifier on TIAGE")
    parser.add_argument("--tiage_root", type=str, default="../tiage",
                        help="Path to TIAGE repository root")
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="Base model to use")
    parser.add_argument("--output_dir", type=str, default="models/topic_shift_roberta",
                        help="Output directory for trained model")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--use_mock_data", action="store_true",
                        help="Use mock data instead of real TIAGE (for testing)")

    args = parser.parse_args()

    train_topic_shift_classifier(
        tiage_root=args.tiage_root,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        use_mock_data=args.use_mock_data,
    )


if __name__ == "__main__":
    main()
