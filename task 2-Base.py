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
# 2. Load & Normalize Dataset
# -----------------------------
def normalize_labels(ds):
    """Ensure evasion_label is always a LIST, even if dataset stores a string."""
    def fix(example):
        lab = example["evasion_label"]

        if isinstance(lab, list):
            return {"evasion_label": lab}

        if lab is None or str(lab).strip() == "":
            return {"evasion_label": ["Other"]}

        if "," in lab:
            labs = [x.strip() for x in lab.split(",") if x.strip()]
            return {"evasion_label": labs}

        return {"evasion_label": [lab]}

    return ds.map(fix)


def load_evasion_dataset(validation_fraction=0.1, seed=42):
    dataset = load_dataset("ailsntua/QEvasion")

    train_full = dataset["train"].shuffle(seed=seed)
    val_size = int(len(train_full) * validation_fraction)

    train = train_full.select(range(len(train_full) - val_size))
    val = train_full.select(range(len(train_full) - val_size, len(train_full)))
    test = dataset["test"]

    # normalize format
    train = normalize_labels(train)
    val   = normalize_labels(val)
    test  = normalize_labels(test)

    return train, val, test



# -----------------------------
# 3. Label Mapping (Single-Label)
# -----------------------------
def build_label_mappings(train_dataset):
    """Use ONLY the first label as the primary class."""
    primary_labels = [labs[0] for labs in train_dataset["evasion_label"]]

    unique_labels = sorted(set(primary_labels))
    print("Detected evasion labels:", unique_labels)

    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    return label2id, id2label, len(unique_labels)



# -----------------------------
# 4. Tokenization + Label Encoding
# -----------------------------
def tokenize_and_encode(train, val, test, label2id, model_name="roberta-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["interview_question"],
            batch["interview_answer"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    train = train.map(tokenize_fn, batched=True)
    val   = val.map(tokenize_fn, batched=True)
    test  = test.map(tokenize_fn, batched=True)

    # encode FIRST LABEL ONLY
    def encode_label(batch):
        return {"labels": [label2id[labs[0]] for labs in batch["evasion_label"]]}

    train = train.map(encode_label, batched=True)
    val   = val.map(encode_label, batched=True)

    # Formats
    train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return tokenizer, train, val, test



# -----------------------------
# 5. DataLoaders
# -----------------------------
def create_dataloaders(train, val, test, train_bs=16, eval_bs=32):
    return (
        DataLoader(train, batch_size=train_bs, shuffle=True),
        DataLoader(val, batch_size=eval_bs, shuffle=False),
        DataLoader(test, batch_size=eval_bs, shuffle=False),
    )



# -----------------------------
# 6. Training (CrossEntropy)
# -----------------------------
def train_evasion_model(
    model_name="roberta-base",
    num_epochs=5,
    train_batch_size=16,
    eval_batch_size=32,
    lr=2e-5,
    seed=42
):
    set_seed(seed)

    # Load & prep
    train, val, test = load_evasion_dataset(seed=seed)
    label2id, id2label, num_labels = build_label_mappings(train)

    tokenizer, train, val, test = tokenize_and_encode(train, val, test, label2id, model_name)

    train_loader, val_loader, test_loader = create_dataloaders(
        train, val, test, train_batch_size, eval_batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_f1 = 0
    best_state = None

    # -------------------------
    # Train Loop
    # -------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for batch in train_loader:
            inp = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            lbl = batch["labels"].to(device)

            logits = model(inp, attention_mask=att).logits
            loss = loss_fn(logits, lbl)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"\nEpoch {epoch} | Train Loss = {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        preds_all, labels_all = [], []

        with torch.no_grad():
            for batch in val_loader:
                inp = batch["input_ids"].to(device)
                att = batch["attention_mask"].to(device)
                lbl = batch["labels"].cpu().numpy()

                logits = model(inp, attention_mask=att).logits
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

                labels_all.extend(lbl)
                preds_all.extend(preds)

        f1 = f1_score(labels_all, preds_all, average="macro")
        print(f"Val Macro-F1 = {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            print("🔥 New best model!")

    # Save best
    model.load_state_dict(best_state)
    torch.save(best_state, "model_task2_best.pt")
    print("\nBest model saved → model_task2_best.pt")

    # Final evaluation
    model.eval()
    preds_all, labels_all = [], []

    with torch.no_grad():
        for batch in val_loader:
            inp = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)
            lbl = batch["labels"].cpu().numpy()

            logits = model(inp, attention_mask=att).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            labels_all.extend(lbl)
            preds_all.extend(preds)

    print("\n=== FINAL VALIDATION REPORT ===")
    print(classification_report(labels_all, preds_all, target_names=[id2label[i] for i in range(num_labels)]))

    return model, tokenizer, label2id, id2label, test_loader



# -----------------------------
# 7. Predict & Save Submission
# -----------------------------
def predict_and_save(model, test_loader, id2label, output_path="prediction"):
    """Save exactly ONE label per row — SemEval submission format."""
    device = next(model.parameters()).device
    model.eval()

    preds_out = []

    with torch.no_grad():
        for batch in test_loader:
            inp = batch["input_ids"].to(device)
            att = batch["attention_mask"].to(device)

            logits = model(inp, attention_mask=att).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            for p in preds:
                preds_out.append(id2label[p])

    # Save WITHOUT extension
    with open(output_path, "w", encoding="utf-8") as f:
        for label in preds_out:
            f.write(label + "\n")




# -----------------------------
# 8. MAIN
# -----------------------------
if __name__ == "__main__":
    model, tokenizer, label2id, id2label, test_loader = train_evasion_model(
        num_epochs=8,
        train_batch_size=16,
        eval_batch_size=32,
        lr=2e-5,
        seed=42
    )
    predict_and_save(model, test_loader, id2label)
