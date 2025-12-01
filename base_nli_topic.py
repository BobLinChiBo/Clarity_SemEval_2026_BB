# base_nli_topic.py
"""
RoBERTa + NLI + Topic Shift feature model for CLARITY Task 1:

- Uses precomputed NLI features (p_contra, p_neutral, p_entail)
- Uses precomputed topic shift feature (p_topic_shift)
  saved by `precompute_topic_shift_features.py` in data/qevasion_with_nli_topic
- Uses RoBERTa-base as the encoder
- Concatenates [CLS] embedding with 4-dim signal features (NLI 3 + topic shift 1)
- Trains on clarity_label (Clear Reply / Ambivalent / Clear Non-Reply)
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from sklearn.metrics import classification_report, f1_score


# --------------------------
# Model definition
# --------------------------

class RobertaWithSignals(nn.Module):
    """RoBERTa with NLI + Topic Shift signal features."""
    def __init__(self, model_name: str, num_labels: int, signal_dim: int = 4, dropout_prob: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size + signal_dim, num_labels)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        signal_feats=None,
        labels=None,
    ):
        """
        Args:
            signal_feats: [batch_size, 4] tensor with [p_contra, p_neutral, p_entail, p_topic_shift]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # RoBERTa doesn't have pooler_output, use CLS token
        pooled = outputs.last_hidden_state[:, 0]

        if signal_feats is not None:
            x = torch.cat([pooled, signal_feats], dim=-1)
        else:
            x = pooled

        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# --------------------------
# Data preparation
# --------------------------

def load_qevasion_with_signals(data_dir: str = "data/qevasion_with_nli_topic"):
    path = Path(data_dir)
    if not path.exists():
        raise RuntimeError(
            f"Cannot find '{data_dir}'. "
            f"Please run these steps first:\n"
            f"  1. python precompute_nli_features.py\n"
            f"  2. python train_topic_shift_tiage.py --use_mock_data\n"
            f"  3. python precompute_topic_shift_features.py"
        )
    return load_from_disk(str(path))


def build_label_mapping(train_split) -> Tuple[Dict[str, int], Dict[int, str]]:
    labels = sorted(list(set(train_split["clarity_label"])))
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    print("Detected clarity labels:", labels)
    return label2id, id2label


def tokenize_dataset(dataset, tokenizer, label2id, max_length: int = 256):
    def encode_batch(batch):
        enc = tokenizer(
            batch["interview_question"],
            batch["interview_answer"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # map label string to id
        labels = [label2id[lbl] for lbl in batch["clarity_label"]]
        enc["labels"] = labels

        # keep NLI features as separate float columns
        enc["p_contra"] = batch["p_contra"]
        enc["p_neutral"] = batch["p_neutral"]
        enc["p_entail"] = batch["p_entail"]

        # NEW: keep topic shift feature
        enc["p_topic_shift"] = batch.get("p_topic_shift", [0.0] * len(labels))

        return enc

    encoded = dataset.map(encode_batch, batched=True)

    # Torch format columns (RoBERTa doesn't use token_type_ids)
    cols = ["input_ids", "attention_mask", "labels",
            "p_contra", "p_neutral", "p_entail", "p_topic_shift"]

    encoded.set_format(type="torch", columns=cols)
    return encoded


# --------------------------
# Training & evaluation loops
# --------------------------

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    log_every: int = 100,
):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # build [B, 4] signal feature tensor: NLI (3) + topic shift (1)
        signal_feats = torch.stack(
            [batch["p_contra"], batch["p_neutral"], batch["p_entail"], batch["p_topic_shift"]],
            dim=-1,
        ).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            signal_feats=signal_feats,
            labels=labels,
        )

        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (step + 1) % log_every == 0:
            avg_loss = total_loss / (step + 1)
            print(f"  [train] step {step+1}/{len(dataloader)}, loss = {avg_loss:.4f}")

    return total_loss / max(1, len(dataloader))


def evaluate(
    model,
    dataloader,
    device,
    id2label: Dict[int, str],
):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            signal_feats = torch.stack(
                [batch["p_contra"], batch["p_neutral"], batch["p_entail"], batch["p_topic_shift"]],
                dim=-1,
            ).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                signal_feats=signal_feats,
                labels=None,
            )
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print("Validation classification report:")
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=target_names,
            digits=3,
        )
    )
    print(f"Validation macro F1: {macro_f1:.4f}")
    return macro_f1


# --------------------------
# Top-level train function
# --------------------------

def train_clarity_with_signals(
    model_name: str = "roberta-base",
    data_dir: str = "data/qevasion_with_nli_topic",
    num_epochs: int = 3,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    seed: int = 42,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading QEvasion dataset with signal features from {data_dir} ...")
    dataset = load_qevasion_with_signals(data_dir=data_dir)

    # Train/val split from official train split
    print("Splitting train/validation (90/10)...")
    train_full = dataset["train"].shuffle(seed=seed)
    val_size = int(len(train_full) * 0.1)

    train_ds = train_full.select(range(len(train_full) - val_size))
    val_ds = train_full.select(range(len(train_full) - val_size, len(train_full)))
    test_ds = dataset["test"]

    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")

    # Label mapping
    label2id, id2label = build_label_mapping(train_full)

    # Tokenizer for RoBERTa
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Tokenizing train/val/test and attaching NLI features...")
    train_enc = tokenize_dataset(train_ds, tokenizer, label2id, max_length=max_length)
    val_enc = tokenize_dataset(val_ds, tokenizer, label2id, max_length=max_length)
    test_enc = tokenize_dataset(test_ds, tokenizer, label2id, max_length=max_length)

    train_loader = DataLoader(train_enc, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_enc, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_enc, batch_size=eval_batch_size, shuffle=False)

    # Build model
    num_labels = len(label2id)
    print(f"Initializing RobertaWithSignals with backbone '{model_name}', num_labels={num_labels}")
    model = RobertaWithSignals(model_name=model_name, num_labels=num_labels, signal_dim=4)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    total_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            log_every=100,
        )
        print(f"[Epoch {epoch}] Train loss: {train_loss:.4f}")

        val_f1 = evaluate(model, val_loader, device, id2label)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"[Epoch {epoch}] New best macro F1 = {val_f1:.4f}, saving best model in memory.")

    if best_state_dict is not None:
        print(f"\nLoading best model state (macro F1 = {best_val_f1:.4f}) for final evaluation and prediction...")
        model.load_state_dict(best_state_dict)

    # Final evaluation on validation (just to print again)
    _ = evaluate(model, val_loader, device, id2label)

    return model, tokenizer, label2id, id2label, test_loader


# --------------------------
# Prediction on test split
# --------------------------

def predict_and_save(
    model,
    test_loader,
    id2label: Dict[int, str],
    output_path: str = "clarity_predictions_nli_topic.txt",
):
    device = next(model.parameters()).device
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            signal_feats = torch.stack(
                [batch["p_contra"], batch["p_neutral"], batch["p_entail"], batch["p_topic_shift"]],
                dim=-1,
            ).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                signal_feats=signal_feats,
                labels=None,
            )
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())

    labels_str = [id2label[i] for i in all_preds]

    with open(output_path, "w", encoding="utf-8") as f:
        for lbl in labels_str:
            f.write(lbl + "\n")

    print(f"Saved test predictions with signal features to '{output_path}'.")


# --------------------------
# Main
# --------------------------

if __name__ == "__main__":
    model, tokenizer, label2id, id2label, test_loader = train_clarity_with_signals(
        model_name="roberta-base",
        data_dir="data/qevasion_with_nli_topic",
        num_epochs=3,
        train_batch_size=16,
        eval_batch_size=32,
        learning_rate=2e-5,
        max_length=256,
        seed=42,
    )

    predict_and_save(model, test_loader, id2label, output_path="clarity_predictions_nli_topic.txt")
