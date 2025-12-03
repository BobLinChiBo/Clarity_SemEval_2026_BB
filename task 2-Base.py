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
def load_evasion_dataset(validation_fraction: float = 0.1, seed: int = 42):
    dataset = load_dataset("ailsntua/QEvasion")

    train_full = dataset["train"].shuffle(seed=seed)
    val_size = int(len(train_full) * validation_fraction)

    train_dataset = train_full.select(range(len(train_full) - val_size))
    val_dataset = train_full.select(range(len(train_full) - val_size, len(train_full)))
    test_dataset = dataset["test"]

    return train_dataset, val_dataset, test_dataset


# -----------------------------
# 3. Prepare label mappings
# -----------------------------
def build_label_mappings(train_dataset):
    """
    Build mapping for MULTI-LABEL evasion categories.
    """
    all_labels = []

    # Flatten list of lists
    for labels in train_dataset["evasion_label"]:
        all_labels.extend(labels)

    unique_labels = sorted(set(all_labels))
    print("Detected evasion labels:", unique_labels)

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    num_labels = len(unique_labels)

    return label2id, id2label, num_labels

 
# -----------------------------
# 4. Tokenization & label encoding
# -----------------------------
def tokenize_and_encode(train_dataset, val_dataset, test_dataset, label2id, model_name="roberta-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["interview_question"],
            examples["interview_answer"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Correct multi-label encoding
    def encode_multilabels(examples):
        # Build a multi-hot vector for each example
        multi_hot = []
        for label_list in examples["evasion_label"]:
            vec = [0] * len(label2id)
            for lab in label_list:
                vec[label2id[lab]] = 1
            multi_hot.append(vec)
        return {"labels": multi_hot}

    train_dataset = train_dataset.map(encode_multilabels, batched=True)
    val_dataset = val_dataset.map(encode_multilabels, batched=True)

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
# 6. Training loop (Subtask 2)
# -----------------------------
def train_clarity_model(
    model_name="roberta-base",
    num_epochs=3,
    train_batch_size=16,
    eval_batch_size=32,
    learning_rate=2e-5,
    seed=42,
):

    set_seed(seed)

    # Load dataset
    train_dataset, val_dataset, test_dataset = load_evasion_dataset(
        validation_fraction=0.1, seed=seed
    )

    # Prepare label mappings
    label2id, id2label, num_labels = build_label_mappings(train_dataset)

    # Tokenization + multi-hot label encoding
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

    # MULTI-LABEL MODEL
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification"
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_f1 = 0.0
    best_model_state = None

    # -------------------------
    # Training
    # -------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits

            loss = loss_fn(logits, batch["labels"].float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"\nEpoch {epoch} | Train Loss = {total_loss / len(train_loader):.4f}")

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].cpu().numpy()
                all_labels.append(labels)

                batch = {k: v.to(device) for k, v in batch.items()}

                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                ).logits

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_preds.append(preds)

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        val_f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"Val Macro-F1 = {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print("🔥 New best model!")

    # -------------------------
    # Save Best Model
    # -------------------------
    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), "model_task2_best.pt")
        print("\nSaved best model as model_task2_best.pt")

    # -------------------------
    # Final validation evaluation
    # -------------------------
    print("\n=== FINAL VALIDATION REPORT ===")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].cpu().numpy()
            all_labels.append(labels)

            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            ).logits

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_preds.append(preds)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    final_f1 = f1_score(all_labels, all_preds, average="macro")
    print("Final Macro-F1:", final_f1)
    print(classification_report(all_labels, all_preds, target_names=[id2label[i] for i in range(num_labels)]))

    return model, tokenizer, label2id, id2label, test_loader


def predict_and_save(model, test_loader, id2label, output_path="evasion_predictions.txt"):
    device = next(model.parameters()).device
    model.eval()

    all_pred_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            logits = outputs.logits

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            # Convert multi-hot → list of labels
            for row in preds:
                labels_for_sample = [id2label[i] for i, v in enumerate(row) if v == 1]

                # If no label predicted → assign "Other"
                if len(labels_for_sample) == 0:
                    labels_for_sample = ["Other"]

                # Format as comma-separated (SemEval expected)
                all_pred_labels.append(", ".join(labels_for_sample))

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_pred_labels:
            f.write(line + "\n")

    print(f"Saved test predictions to: {output_path}")


if __name__ == "__main__":
    model, tokenizer, label2id, id2label, test_loader = train_clarity_model(
        model_name="roberta-base",
        num_epochs=5,
        train_batch_size=16,
        eval_batch_size=32,
        learning_rate=2e-5,
        seed=42,
    )

    predict_and_save(model, test_loader, id2label, output_path="evasion_predictions.txt")
