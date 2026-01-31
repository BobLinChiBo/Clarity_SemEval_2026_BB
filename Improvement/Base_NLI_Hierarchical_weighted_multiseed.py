import json
import math
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import f1_score, accuracy_score, classification_report

# ============================================================
# 1) Reproducibility helper
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2) Model (RobertaNLIHier)
# ============================================================
class RobertaNLIHier(nn.Module):
    """
    Shared RoBERTa + gated NLI features, with 2-stage hierarchical heads.

    Stage 1: Reply (Clear Reply) vs Non-Reply-ish (Ambivalent + Clear Non-Reply)
    Stage 2: Ambivalent vs Clear Non-Reply (used only if Stage 1 says Non-Reply-ish)
    """

    def __init__(
        self,
        model_name: str,
        nli_feat_dim: int = 3,
        nli_hidden_dim: int = 32,
        dropout_prob: float = 0.1,
        stage1_class_weights: Optional[torch.Tensor] = None,  # [2]
        stage2_class_weights: Optional[torch.Tensor] = None,  # [2]
        lambda2: float = 0.7,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.nli_mlp = nn.Sequential(
            nn.Linear(nli_feat_dim, nli_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(nli_hidden_dim, nli_hidden_dim),
            nn.ReLU(),
        )

        self.gate_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_prob)

        feat_dim = hidden_size + nli_hidden_dim
        self.stage1_head = nn.Linear(feat_dim, 2)
        self.stage2_head = nn.Linear(feat_dim, 2)

        # keep weights on device via buffers
        if stage1_class_weights is not None:
            self.register_buffer("stage1_class_weights", stage1_class_weights)
        else:
            self.stage1_class_weights = None

        if stage2_class_weights is not None:
            self.register_buffer("stage2_class_weights", stage2_class_weights)
        else:
            self.stage2_class_weights = None

        self.lambda2 = float(lambda2)

    def encode(self, input_ids, attention_mask=None, nli_feats=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS

        if nli_feats is not None:
            nli_repr = self.nli_mlp(nli_feats)
            gate = torch.sigmoid(self.gate_layer(pooled))  # [B,1]
            nli_repr = gate * nli_repr
            x = torch.cat([pooled, nli_repr], dim=-1)
        else:
            x = pooled

        return self.dropout(x)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        nli_feats=None,
        stage1_labels=None,  # [B] in {0,1}
        stage2_labels=None,  # [B] in {0,1}
        stage2_mask=None,    # [B] bool
    ):
        x = self.encode(input_ids, attention_mask, nli_feats)
        logits1 = self.stage1_head(x)  # [B,2]
        logits2 = self.stage2_head(x)  # [B,2]

        loss1 = None
        loss2 = None
        loss = None

        if stage1_labels is not None:
            loss_fn1 = nn.CrossEntropyLoss(weight=self.stage1_class_weights) if self.stage1_class_weights is not None else nn.CrossEntropyLoss()
            loss1 = loss_fn1(logits1, stage1_labels)

        if stage2_labels is not None and stage2_mask is not None:
            idx = stage2_mask.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                loss_fn2 = nn.CrossEntropyLoss(weight=self.stage2_class_weights) if self.stage2_class_weights is not None else nn.CrossEntropyLoss()
                loss2 = loss_fn2(logits2[idx], stage2_labels[idx])
            else:
                loss2 = torch.tensor(0.0, device=logits1.device)

        if loss1 is not None and loss2 is not None:
            loss = loss1 + self.lambda2 * loss2
        elif loss1 is not None:
            loss = loss1
        elif loss2 is not None:
            loss = self.lambda2 * loss2

        return {"loss": loss, "loss1": loss1, "loss2": loss2, "logits1": logits1, "logits2": logits2}


# ============================================================
# 3) Data helpers
# ============================================================
def load_qevasion_with_nli(data_dir: str = "data/qevasion_with_nli"):
    path = Path(data_dir)
    if not path.exists():
        raise RuntimeError(f"Cannot find '{data_dir}'. Run precompute + save_to_disk first.")
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
        enc["labels"] = [label2id[lbl] for lbl in batch["clarity_label"]]
        enc["p_contra"] = batch["p_contra"]
        enc["p_neutral"] = batch["p_neutral"]
        enc["p_entail"] = batch["p_entail"]
        return enc

    encoded = dataset.map(encode_batch, batched=True)
    cols = ["input_ids", "attention_mask", "labels", "p_contra", "p_neutral", "p_entail"]
    encoded.set_format(type="torch", columns=cols)
    return encoded


# ============================================================
# 4) Stage targets + weights
# ============================================================
def make_stage_targets(labels_3way: torch.Tensor, id2label: Dict[int, str]):
    label_str = [id2label[i.item()] for i in labels_3way]
    stage1 = torch.tensor([1 if s == "Clear Reply" else 0 for s in label_str], device=labels_3way.device)
    stage2 = torch.tensor([1 if s == "Clear Non-Reply" else 0 for s in label_str], device=labels_3way.device)
    stage2_mask = (stage1 == 0)
    return stage1.long(), stage2.long(), stage2_mask


def compute_stage_class_weights(train_split, label2id, id2label, device, cap: float = 3.0, use_sqrt: bool = True):
    from collections import Counter

    y = torch.tensor([label2id[lbl] for lbl in train_split["clarity_label"]], dtype=torch.long)

    stage1_cpu = torch.tensor([1 if id2label[i.item()] == "Clear Reply" else 0 for i in y], dtype=torch.long)
    stage2_cpu = torch.tensor([1 if id2label[i.item()] == "Clear Non-Reply" else 0 for i in y], dtype=torch.long)
    stage2_mask_cpu = (stage1_cpu == 0)

    c1 = Counter(stage1_cpu.tolist())
    c2 = Counter(stage2_cpu[stage2_mask_cpu].tolist())

    def inv_freq(counter, n_classes=2):
        total = sum(counter.values())
        weights = []
        for cls in range(n_classes):
            denom = max(1, counter.get(cls, 0))
            w = total / (n_classes * denom)
            if use_sqrt:
                w = math.sqrt(w)
            if cap is not None:
                w = min(w, cap)
            weights.append(w)
        return torch.tensor(weights, dtype=torch.float, device=device)

    w1 = inv_freq(c1, 2)
    w2 = inv_freq(c2, 2)

    print("\nStage1 weights (NonReplyish=0, Reply=1):", w1.tolist(), "counts:", dict(c1))
    print("Stage2 weights (Amb=0, CNR=1):", w2.tolist(), "counts:", dict(c2))
    return w1, w2


# ============================================================
# 5) Train / eval utilities (single model)
# ============================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, device, id2label, log_every=100):
    model.train()
    total_loss = 0.0

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_3 = batch["labels"].to(device)

        nli_feats = torch.stack([batch["p_contra"], batch["p_neutral"], batch["p_entail"]], dim=-1).to(device)
        stage1, stage2, stage2_mask = make_stage_targets(labels_3, id2label)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            nli_feats=nli_feats,
            stage1_labels=stage1,
            stage2_labels=stage2,
            stage2_mask=stage2_mask,
        )

        loss = out["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += float(loss.item())

        if (step + 1) % log_every == 0:
            print(f"  [train] step {step+1}/{len(dataloader)}, loss={total_loss/(step+1):.4f}")

    return total_loss / max(1, len(dataloader))


@torch.no_grad()
def predict_thresholded(model, dataloader, device, id2label: Dict[int, str], t_reply: float, t_cnr: float) -> List[int]:
    model.eval()
    inv = {v: k for k, v in id2label.items()}  # label string -> id
    preds: List[int] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        nli_feats = torch.stack([batch["p_contra"], batch["p_neutral"], batch["p_entail"]], dim=-1).to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, nli_feats=nli_feats)
        p1 = F.softmax(out["logits1"], dim=-1)  # [B,2]
        p2 = F.softmax(out["logits2"], dim=-1)  # [B,2]

        p_reply = p1[:, 1]
        p_cnr = p2[:, 1]

        for i in range(p_reply.size(0)):
            if float(p_reply[i].item()) > t_reply:
                preds.append(inv["Clear Reply"])
            else:
                preds.append(inv["Clear Non-Reply"] if float(p_cnr[i].item()) > t_cnr else inv["Ambivalent"])

    return preds


@torch.no_grad()
def evaluate_thresholded(model, dataloader, device, id2label, t_reply, t_cnr, title="Eval") -> Dict[str, float]:
    labels = []
    for batch in dataloader:
        labels.extend(batch["labels"].cpu().tolist())

    preds = predict_thresholded(model, dataloader, device, id2label, t_reply=t_reply, t_cnr=t_cnr)

    macro_f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)

    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print(f"\n--- {title} ---")
    print(f"t_reply={t_reply:.2f}, t_cnr={t_cnr:.2f}")
    print(classification_report(labels, preds, target_names=target_names, digits=3))
    print(f"{title} macro_f1: {macro_f1:.4f} | acc: {acc:.4f}")

    return {"macro_f1": macro_f1, "accuracy": acc}


def tune_thresholds_on_validation(model, val_loader, device, id2label, reply_grid=None, cnr_grid=None) -> Tuple[float, float, float]:
    if reply_grid is None:
        reply_grid = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
    if cnr_grid is None:
        cnr_grid = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]

    gold = []
    for batch in val_loader:
        gold.extend(batch["labels"].cpu().tolist())

    best_f1, best_t_reply, best_t_cnr = -1.0, None, None

    for t_reply in reply_grid:
        for t_cnr in cnr_grid:
            preds = predict_thresholded(model, val_loader, device, id2label, t_reply=t_reply, t_cnr=t_cnr)
            f1 = f1_score(gold, preds, average="macro")
            if f1 > best_f1:
                best_f1, best_t_reply, best_t_cnr = f1, t_reply, t_cnr

    print(f"\n[Single Model] Best thresholds: t_reply={best_t_reply:.2f}, t_cnr={best_t_cnr:.2f}, macro_f1={best_f1:.4f}")
    return float(best_t_reply), float(best_t_cnr), float(best_f1)


# ============================================================
# 6) Train ONE seed (but keep split fixed with split_seed)
# ============================================================
def train_clarity_with_nli_hier(
    model_name: str = "roberta-base",
    data_dir: str = "data/qevasion_with_nli",
    num_epochs: int = 10,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    learning_rate: float = 2e-5,
    max_length: int = 256,
    seed: int = 42,
    split_seed: int = 42,
    use_stage_weights: bool = True,
    lambda2: float = 0.7,
    model_nickname: str = "roberta_hier",
):
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = load_qevasion_with_nli(data_dir=data_dir)

    print(f"Splitting train/validation (90/10) using split_seed={split_seed} ...")
    train_full = dataset["train"].shuffle(seed=split_seed)
    val_size = int(len(train_full) * 0.1)

    train_ds = train_full.select(range(len(train_full) - val_size))
    val_ds = train_full.select(range(len(train_full) - val_size, len(train_full)))
    test_ds = dataset["test"]

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | QEvasion Test: {len(test_ds)}")

    label2id, id2label = build_label_mapping(train_full)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_enc = tokenize_dataset(train_ds, tokenizer, label2id, max_length=max_length)
    val_enc   = tokenize_dataset(val_ds, tokenizer, label2id, max_length=max_length)
    test_enc  = tokenize_dataset(test_ds, tokenizer, label2id, max_length=max_length)

    train_loader = DataLoader(train_enc, batch_size=train_batch_size, shuffle=True)
    val_loader   = DataLoader(val_enc, batch_size=eval_batch_size, shuffle=False)
    test_loader  = DataLoader(test_enc, batch_size=eval_batch_size, shuffle=False)

    w1, w2 = None, None
    if use_stage_weights:
        w1, w2 = compute_stage_class_weights(train_ds, label2id, id2label, device, cap=3.0, use_sqrt=True)

    model = RobertaNLIHier(
        model_name=model_name,
        nli_feat_dim=3,
        stage1_class_weights=w1,
        stage2_class_weights=w2,
        lambda2=lambda2,
    ).to(device)

    encoder_params = list(model.encoder.parameters())
    other_params = (
        list(model.nli_mlp.parameters())
        + list(model.gate_layer.parameters())
        + list(model.stage1_head.parameters())
        + list(model.stage2_head.parameters())
    )

    optimizer = AdamW(
        [
            {"params": encoder_params, "lr": learning_rate},
            {"params": other_params, "lr": learning_rate * 5},
        ]
    )

    total_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} (seed={seed}) =====")
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, id2label, log_every=100)
        print(f"[Epoch {epoch}] Train loss: {tr_loss:.4f}")

        metrics = evaluate_thresholded(model, val_loader, device, id2label, t_reply=0.70, t_cnr=0.70, title="Val (default thr)")
        val_f1 = metrics["macro_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            print(f"[Epoch {epoch}] New best default-thr macro F1={best_val_f1:.4f}")

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = f"model_hier_seed{seed}_{model_nickname}.pt"
    torch.save(best_state, ckpt)
    print(f"\nSaved checkpoint: {ckpt}")

    # tune thresholds for THIS single model (optional; ensemble will retune anyway)
    best_t_reply, best_t_cnr, best_thr_f1 = tune_thresholds_on_validation(model, val_loader, device, id2label)

    # save summary
    out_json = f"results_hier_seed{seed}_{model_nickname}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": seed,
                "split_seed": split_seed,
                "best_val_macro_f1_default_thr": best_val_f1,
                "best_thresholds_single_model": {"t_reply": best_t_reply, "t_cnr": best_t_cnr, "val_macro_f1": best_thr_f1},
            },
            f,
            indent=2,
        )
    print(f"Saved summary: {out_json}")

    return model, tokenizer, label2id, id2label, val_loader, test_loader


# ============================================================
# 7) Multi-seed ensemble helpers (AVERAGE logits1/logits2)
# ============================================================
@torch.no_grad()
def ensemble_logits_hier(models: List[RobertaNLIHier], dataloader: DataLoader, device: torch.device):
    for m in models:
        m.eval()

    logits1_all = []
    logits2_all = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        nli_feats = torch.stack([batch["p_contra"], batch["p_neutral"], batch["p_entail"]], dim=-1).to(device)

        per_model_l1 = []
        per_model_l2 = []
        for m in models:
            out = m(input_ids=input_ids, attention_mask=attention_mask, nli_feats=nli_feats)
            per_model_l1.append(out["logits1"].float().cpu())
            per_model_l2.append(out["logits2"].float().cpu())

        logits1_all.append(torch.stack(per_model_l1, dim=0).mean(dim=0))  # [B,2]
        logits2_all.append(torch.stack(per_model_l2, dim=0).mean(dim=0))  # [B,2]

    mean_logits1 = torch.cat(logits1_all, dim=0)  # [N,2]
    mean_logits2 = torch.cat(logits2_all, dim=0)  # [N,2]
    return mean_logits1, mean_logits2


def decode_thresholded_from_logits(mean_logits1, mean_logits2, id2label, t_reply, t_cnr):
    inv = {v: k for k, v in id2label.items()}  # label string -> id

    p1 = F.softmax(mean_logits1, dim=-1)
    p2 = F.softmax(mean_logits2, dim=-1)

    p_reply = p1[:, 1]
    p_cnr = p2[:, 1]

    preds = []
    for i in range(p_reply.size(0)):
        if float(p_reply[i].item()) > t_reply:
            preds.append(inv["Clear Reply"])
        else:
            preds.append(inv["Clear Non-Reply"] if float(p_cnr[i].item()) > t_cnr else inv["Ambivalent"])
    return np.array(preds, dtype=np.int64)


def tune_thresholds_ensemble(mean_logits1, mean_logits2, gold_ids, id2label, reply_grid=None, cnr_grid=None):
    if reply_grid is None:
        reply_grid = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
    if cnr_grid is None:
        cnr_grid = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]

    best_f1, best_t_reply, best_t_cnr = -1.0, None, None
    for t_reply in reply_grid:
        for t_cnr in cnr_grid:
            preds = decode_thresholded_from_logits(mean_logits1, mean_logits2, id2label, t_reply, t_cnr)
            f1 = f1_score(gold_ids, preds, average="macro")
            if f1 > best_f1:
                best_f1, best_t_reply, best_t_cnr = f1, t_reply, t_cnr

    print(f"\n[Ensemble] Best thresholds: t_reply={best_t_reply:.2f}, t_cnr={best_t_cnr:.2f}, macro_f1={best_f1:.4f}")
    return float(best_t_reply), float(best_t_cnr), float(best_f1)


def save_codabench_prediction_file(pred_ids, id2label, out_path="prediction"):
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for pid in pred_ids:
            f.write(id2label[int(pid)] + "\n")
    print(f"[OK] Wrote {len(pred_ids)} lines to: {out_path.resolve()}")


# ============================================================
# 8) MAIN: multi-seed training + ensemble + save "prediction"
# ============================================================
def main():
    SEEDS = [21, 42, 84]
    SPLIT_SEED = 42
    NUM_EPOCHS = 10

    MODEL_NAME = "roberta-base"
    DATA_DIR = "data/qevasion_with_nli"
    MODEL_NICKNAME = "roberta_hier_multiseed"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Seeds:", SEEDS)
    print("Split seed:", SPLIT_SEED)

    models = []
    tokenizer = None
    label2id = None
    id2label = None

    # ---- Train each seed (same split_seed each time!) ----
    for seed in SEEDS:
        print("\n" + "=" * 80)
        print(f"Training seed={seed}")
        print("=" * 80)

        model, tokenizer, label2id, id2label, val_loader, test_loader = train_clarity_with_nli_hier(
            model_name=MODEL_NAME,
            data_dir=DATA_DIR,
            num_epochs=NUM_EPOCHS,
            train_batch_size=16,
            eval_batch_size=32,
            learning_rate=2e-5,
            max_length=256,
            seed=seed,
            split_seed=SPLIT_SEED,
            use_stage_weights=True,
            lambda2=0.7,
            model_nickname=MODEL_NICKNAME,
        )

        models.append(model.to(device))

    # ---- Build gold labels for the FIXED validation set ----
    # easiest: reuse the val_loader we already built (from last seed),
    # and read labels directly from it
    gold_val = []
    for batch in val_loader:
        gold_val.extend(batch["labels"].cpu().numpy().tolist())
    gold_val = np.array(gold_val, dtype=np.int64)

    # ---- Ensemble validation logits ----
    mean_l1_val, mean_l2_val = ensemble_logits_hier(models, val_loader, device)

    # ---- Tune thresholds on ensemble ----
    best_t_reply, best_t_cnr, best_f1 = tune_thresholds_ensemble(
        mean_l1_val, mean_l2_val, gold_val, id2label
    )

    # ---- Report (optional) ----
    val_preds = decode_thresholded_from_logits(mean_l1_val, mean_l2_val, id2label, best_t_reply, best_t_cnr)
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    print("\n[Ensemble] Validation report:")
    print(classification_report(gold_val, val_preds, target_names=target_names, digits=3))
    print("Val macro_f1:", f1_score(gold_val, val_preds, average="macro"))

    # ---- Ensemble test logits + write Codabench file ----
    mean_l1_test, mean_l2_test = ensemble_logits_hier(models, test_loader, device)
    test_preds = decode_thresholded_from_logits(mean_l1_test, mean_l2_test, id2label, best_t_reply, best_t_cnr)

    save_codabench_prediction_file(test_preds, id2label, out_path="prediction")

    # ---- Save summary ----
    with open("results_hier_multiseed_ensemble.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seeds": SEEDS,
                "split_seed": SPLIT_SEED,
                "num_epochs": NUM_EPOCHS,
                "best_thresholds": {"t_reply": best_t_reply, "t_cnr": best_t_cnr, "val_macro_f1": best_f1},
            },
            f,
            indent=2,
        )
    print("[OK] Saved summary: results_hier_multiseed_ensemble.json")


if __name__ == "__main__":
    main()
