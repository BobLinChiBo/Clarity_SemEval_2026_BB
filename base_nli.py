# base_nli.py
"""
RoBERTa + NLI feature model for CLARITY Task 1 (Level 1 integration):

- Uses precomputed NLI features (p_contra, p_neutral, p_entail)
  saved by `precompute_nli_features.py` in data/qevasion_with_nli
- Uses RoBERTa-base as the encoder (instead of BERT)
- Concatenates [CLS] embedding with 3-dim NLI feature
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

class RobertaWithNLI(nn.Module):
    """
    RoBERTa encoder + NLI sub-network:
    - nli_feats: [B, 3] (p_contra, p_neutral, p_entail)
    - Pass through a 2-layer MLP to get richer representation
    - Use gate to adaptively weight NLI signal based on CLS
    - Concat gated NLI repr with CLS for final classification
    """
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        nli_feat_dim: int = 3,
        nli_hidden_dim: int = 32,
        dropout_prob: float = 0.1,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # NLI sub-network: [B, 3] → [B, nli_hidden_dim]
        self.nli_mlp = nn.Sequential(
            nn.Linear(nli_feat_dim, nli_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(nli_hidden_dim, nli_hidden_dim),
            nn.ReLU(),
        )

        # Gate: CLS decides how much to trust NLI signal
        self.gate_layer = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size + nli_hidden_dim, num_labels)

        # Store class weights for loss computation
        self.register_buffer('class_weights', class_weights)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        nli_feats=None,
        labels=None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # RoBERTa doesn't have pooler_output, use CLS token
        pooled = outputs.last_hidden_state[:, 0]         # [B, hidden_size]

        if nli_feats is not None:
            # [B, 3] → [B, nli_hidden_dim]
            nli_repr = self.nli_mlp(nli_feats)

            # Gate: [B, hidden_size] → [B, 1] → sigmoid → (0,1)
            gate = torch.sigmoid(self.gate_layer(pooled))
            nli_repr = gate * nli_repr                    # broadcast: [B, 1] * [B, nli_hidden_dim]

            x = torch.cat([pooled, nli_repr], dim=-1)     # [B, hidden_size + nli_hidden_dim]
        else:
            x = pooled

        x = self.dropout(x)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            if self.class_weights is not None:
                loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# --------------------------
# Data preparation
# --------------------------

def compute_class_weights(train_split, label2id, device):
    """
    Compute inverse frequency class weights for handling class imbalance.

    Formula: weight[i] = total_samples / (n_classes * count[i])

    Args:
        train_split: Training dataset split with 'clarity_label' field
        label2id: Dictionary mapping label strings to indices
        device: torch device

    Returns:
        torch.Tensor: Class weights tensor of shape [n_classes]
    """
    from collections import Counter

    # Count samples per class
    counter = Counter(train_split["clarity_label"])
    total = sum(counter.values())
    n_classes = len(label2id)

    # Compute weights in label2id order
    weights = []
    for label, idx in sorted(label2id.items(), key=lambda x: x[1]):
        weight = total / (n_classes * counter[label])
        weights.append(weight)

    weights_tensor = torch.tensor(weights, dtype=torch.float, device=device)

    # Print for transparency
    print("\nClass weights (inverse frequency):")
    for label, idx in sorted(label2id.items(), key=lambda x: x[1]):
        print(f"  {label:<20} count={counter[label]:<6} weight={weights[idx]:.4f}")

    return weights_tensor


def load_qevasion_with_nli(data_dir: str = "data/qevasion_with_nli"):
    path = Path(data_dir)
    if not path.exists():
        raise RuntimeError(
            f"Cannot find '{data_dir}'. "
            f"Please run `python precompute_nli_features.py` first."
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

        return enc

    encoded = dataset.map(encode_batch, batched=True)

    # Torch format columns (RoBERTa doesn't use token_type_ids)
    cols = ["input_ids", "attention_mask", "labels",
            "p_contra", "p_neutral", "p_entail"]

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

        # build [B, 3] NLI feature tensor
        nli_feats = torch.stack(
            [batch["p_contra"], batch["p_neutral"], batch["p_entail"]],
            dim=-1,
        ).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            nli_feats=nli_feats,
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
    return_report: bool = False,
):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            nli_feats = torch.stack(
                [batch["p_contra"], batch["p_neutral"], batch["p_entail"]],
                dim=-1,
            ).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                nli_feats=nli_feats,
                labels=None,
            )
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    print("Validation classification report:")
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        digits=3,
    )
    print(report_str)
    print(f"Validation macro F1: {macro_f1:.4f}")

    if return_report:
        return macro_f1, report_str
    return macro_f1


# --------------------------
# Top-level train function
# --------------------------

def train_clarity_with_nli(
    model_name: str = "roberta-base",
    data_dir: str = "data/qevasion_with_nli",
    num_epochs: int = 3,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    seed: int = 42,
    use_class_weights: bool = False,
    model_nickname: str = None,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading QEvasion dataset with NLI features from {data_dir} ...")
    dataset = load_qevasion_with_nli(data_dir=data_dir)

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

    # Compute class weights if requested
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights(train_full, label2id, device)
    else:
        print("\nUsing unweighted CrossEntropyLoss (balanced classes assumed)")

    print(f"Initializing RobertaWithNLI with backbone '{model_name}', num_labels={num_labels}")
    model = RobertaWithNLI(
        model_name=model_name,
        num_labels=num_labels,
        nli_feat_dim=3,
        class_weights=class_weights,
    )
    model.to(device)

    # Layered learning rates: encoder (small) vs. NLI sub-network + classifier (large)
    encoder_params = list(model.encoder.parameters())
    other_params = (
        list(model.nli_mlp.parameters()) +
        list(model.gate_layer.parameters()) +
        list(model.classifier.parameters())
    )

    optimizer = AdamW([
        {"params": encoder_params, "lr": learning_rate},           # 2e-5 for RoBERTa
        {"params": other_params, "lr": learning_rate * 5},         # 1e-4 for sub-network
    ])

    total_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_f1 = 0.0
    best_state_dict = None
    training_history = []  # Store epoch-level metrics

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
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            print(f"[Epoch {epoch}] New best macro F1 = {val_f1:.4f}, saving best model in memory.")

    if best_state_dict is not None:
        print(f"\nLoading best model state (macro F1 = {best_val_f1:.4f}) for final evaluation and prediction...")
        model.load_state_dict(best_state_dict)

        # Save best model checkpoint with unique naming
        # Auto-generate nickname from model_name if not provided
        if model_nickname is None:
            if "deberta" in model_name.lower():
                model_nickname = "deberta"
            elif "roberta" in model_name.lower():
                model_nickname = "roberta"
            elif "bert" in model_name.lower():
                model_nickname = "bert"
            else:
                model_nickname = "model"

        weight_suffix = "weighted" if use_class_weights else "unweighted"
        checkpoint_path = f"model_nli_{weight_suffix}_seed{seed}_{model_nickname}.pt"
        torch.save(best_state_dict, checkpoint_path)
        print(f"Saved best model checkpoint to: {checkpoint_path}")

    # Final evaluation on validation (just to print again)
    final_f1, final_report = evaluate(model, val_loader, device, id2label, return_report=True)

    # Save training summary
    import json
    summary = {
        "model_name": model_name,
        "model_nickname": model_nickname,
        "seed": seed,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "use_class_weights": use_class_weights,
        "best_val_f1": best_val_f1,
        "final_val_f1": final_f1,
        "training_history": training_history,
        "final_classification_report": final_report,
    }

    summary_path = f"results_nli_{weight_suffix}_seed{seed}_{model_nickname}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved training summary to '{summary_path}'")

    return model, tokenizer, label2id, id2label, test_loader


# --------------------------
# Prediction on test split
# --------------------------

def predict_and_save(
    model,
    test_loader,
    id2label: Dict[int, str],
    output_path: str = "clarity_predictions_nli.txt",
):
    device = next(model.parameters()).device
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            nli_feats = torch.stack(
                [batch["p_contra"], batch["p_neutral"], batch["p_entail"]],
                dim=-1,
            ).to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                nli_feats=nli_feats,
                labels=None,
            )
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())

    labels_str = [id2label[i] for i in all_preds]

    with open(output_path, "w", encoding="utf-8") as f:
        for lbl in labels_str:
            f.write(lbl + "\n")

    print(f"Saved test predictions with NLI features to '{output_path}'.")


# --------------------------
# Main
# --------------------------

if __name__ == "__main__":
    model, tokenizer, label2id, id2label, test_loader = train_clarity_with_nli(
        model_name="roberta-base",
        data_dir="data/qevasion_with_nli",
        num_epochs=5,              # Increased from 3 to 5
        train_batch_size=16,
        eval_batch_size=32,
        learning_rate=2e-5,        # Base LR for encoder (sub-network uses 5x)
        max_length=256,
        seed=42,
    )

    predict_and_save(model, test_loader, id2label, output_path="clarity_predictions_nli.txt")
