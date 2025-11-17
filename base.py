import os
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report


# -----------------------------
# 1. Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# 2. Load dataset & splits
# -----------------------------
def load_clarity_dataset(validation_fraction: float = 0.1, seed: int = 42):
    """
    Load ailsntua/QEvasion and create train / val / test splits
    for CLARITY Subtask 1 (clarity-level classification).
    """
    dataset = load_dataset("ailsntua/QEvasion")  # train + test splits

    # Shuffle train and split into train/val
    train_full = dataset["train"].shuffle(seed=seed)
    val_size = int(len(train_full) * validation_fraction)

    train_dataset = train_full.select(range(len(train_full) - val_size))
    val_dataset = train_full.select(range(len(train_full) - val_size, len(train_full)))
    test_dataset = dataset["test"]  # used only for prediction, treated as unlabeled

    return train_dataset, val_dataset, test_dataset


# -----------------------------
# 3. Prepare label mappings
# -----------------------------
def build_label_mappings(train_dataset):
    """
    Build label2id / id2label from the *actual* labels in the dataset.
    This avoids KeyErrors like 'Ambivalent' vs 'Ambivalent Reply'.
    """
    # Use the clarity_label column directly
    unique_labels = sorted(set(train_dataset["clarity_label"]))
    print("Detected clarity labels:", unique_labels)

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(unique_labels)

    return label2id, id2label, num_labels


# -----------------------------
# 4. Tokenization & label encoding
# -----------------------------
def tokenize_and_encode(train_dataset, val_dataset, test_dataset, label2id, model_name="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        # Question-answer pair
        return tokenizer(
            examples["interview_question"],
            examples["interview_answer"],
            padding="max_length",
            truncation=True,
            max_length=256,  # a bit longer than 128, since answers can be long
        )

    # Add tokenized fields
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Add numeric labels ONLY for train/val
    def add_label_ids(examples):
        return {"labels": [label2id[l] for l in examples["clarity_label"]]}

    train_dataset = train_dataset.map(add_label_ids, batched=True)
    val_dataset = val_dataset.map(add_label_ids, batched=True)

    # For test, we treat it as unlabeled – do NOT touch clarity_label there.
    # Just set the format to return tensors for inputs.
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
    )

    return tokenizer, train_dataset, val_dataset, test_dataset


# -----------------------------
# 5. DataLoaders
# -----------------------------
def create_dataloaders(train_dataset, val_dataset, test_dataset,
                       train_batch_size=16, eval_batch_size=32):

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# -----------------------------
# 6. Training loop (Subtask 1)
# -----------------------------
def train_clarity_model(
    model_name="bert-base-uncased",
    num_epochs=3,
    train_batch_size=16,
    eval_batch_size=32,
    learning_rate=2e-5,
    seed=42,
):

    set_seed(seed)

    # Load and split dataset
    train_dataset, val_dataset, test_dataset = load_clarity_dataset(
        validation_fraction=0.1, seed=seed
    )

    # Build label mappings from train set
    label2id, id2label, num_labels = build_label_mappings(train_dataset)

    # Tokenize and encode labels
    tokenizer, train_dataset, val_dataset, test_dataset = tokenize_and_encode(
        train_dataset, val_dataset, test_dataset, label2id, model_name=model_name
    )

    # DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # -------- Training loop --------
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}  # input_ids, attention_mask, labels

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # -------- Validation --------
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].numpy()
                all_labels.extend(labels)

                batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)

        val_f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Macro-F1: {val_f1:.4f}")
        print(classification_report(all_labels, all_preds, target_names=[id2label[i] for i in range(num_labels)]))

    return model, tokenizer, label2id, id2label, test_loader


# -----------------------------
# 7. Inference on test & save
# -----------------------------
def predict_and_save(model, test_loader, id2label, output_path="clarity_predictions.txt"):
    device = next(model.parameters()).device
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}  # input_ids, attention_mask

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    # Map numeric predictions back to label names
    pred_labels = [id2label[p] for p in all_preds]

    with open(output_path, "w", encoding="utf-8") as f:
        for label in pred_labels:
            f.write(label + "\n")

    print(f"Saved clarity predictions for test set to: {output_path}")


# -----------------------------
# 8. Main
# -----------------------------
if __name__ == "__main__":
    model, tokenizer, label2id, id2label, test_loader = train_clarity_model(
        model_name="bert-base-uncased",
        num_epochs=3,
        train_batch_size=16,
        eval_batch_size=32,
        learning_rate=2e-5,
        seed=42,
    )

    predict_and_save(model, test_loader, id2label, output_path="clarity_predictions.txt")
