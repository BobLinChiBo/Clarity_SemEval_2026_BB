# precompute_nli_features.py
"""
Compute NLI (entail / neutral / contradiction) probabilities for each
(question, answer) pair in the QEvasion dataset and save a local copy
with extra columns:
    - p_contra
    - p_neutral
    - p_entail

Uses RoBERTa-large-mnli for NLI inference.

Output: data/qevasion_with_nli (HuggingFace Dataset saved to disk)
"""

import torch
from pathlib import Path

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification


NLI_MODEL_NAME = "roberta-large-mnli"  # RoBERTa MNLI NLI model on HF
MAX_LENGTH = 256
BATCH_SIZE = 32


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NLI] Using device: {device}")

    print("[NLI] Loading QEvasion dataset from HuggingFace...")
    dataset = load_dataset("ailsntua/QEvasion")

    print(f"[NLI] Loading NLI model: {NLI_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(device)
    model.eval()

    def add_nli(batch):
        """
        batch: dict with keys including "interview_question", "interview_answer"
        Return dict with lists for p_contra, p_neutral, p_entail
        """
        questions = batch["interview_question"]
        answers = batch["interview_answer"]

        enc = tokenizer(
            answers,                    # premise
            questions,                  # hypothesis
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu()  # shape: [B, 3]

        # For roberta-large-mnli, label order is: contradiction, neutral, entailment
        p_contra = probs[:, 0].tolist()
        p_neutral = probs[:, 1].tolist()
        p_entail = probs[:, 2].tolist()

        return {
            "p_contra": p_contra,
            "p_neutral": p_neutral,
            "p_entail": p_entail,
        }

    new_splits = {}
    for split in dataset.keys():
        print(f"[NLI] Computing NLI features for split '{split}' "
              f"with {len(dataset[split])} examples...")

        # batched=True: will feed BATCH_SIZE examples at a time
        ds_with_nli = dataset[split].map(
            add_nli,
            batched=True,
            batch_size=BATCH_SIZE,
        )
        new_splits[split] = ds_with_nli

    new_dataset = DatasetDict(new_splits)

    out_dir = Path("data/qevasion_with_nli")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[NLI] Saving dataset with NLI features to '{out_dir}'...")
    new_dataset.save_to_disk(str(out_dir))
    print("[NLI] Done.")


if __name__ == "__main__":
    main()
