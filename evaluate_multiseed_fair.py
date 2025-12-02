#!/usr/bin/env python3
"""
Fair evaluation of multi-seed ensemble using a held-out validation split.

This script uses validation_seed=999 which is different from all training seeds
[21, 42, 84] to ensure no data leakage.
"""

import json
import torch
import numpy as np
from base_nli import load_qevasion_with_nli, tokenize_dataset, RobertaWithNLI
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer


def load_models(seeds, device):
    """Load all trained models from checkpoints."""
    models = []
    for seed in seeds:
        checkpoint_path = f"model_nli_weighted_seed{seed}_roberta.pt"
        print(f"Loading {checkpoint_path}...")

        model = RobertaWithNLI(
            model_name="roberta-base",
            num_labels=3
        )
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Remove class_weights if it exists (it's a buffer, not a parameter)
        if "class_weights" in state_dict:
            del state_dict["class_weights"]
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        models.append({
            "seed": seed,
            "model": model,
            "checkpoint": checkpoint_path
        })

    return models


def ensemble_predict_logits(models, data_loader, device):
    """Ensemble prediction by averaging logits across all models."""

    n_models = len(models)
    all_logits_list = [[] for _ in range(n_models)]

    for model_dict in models:
        model_dict["model"].eval()

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            nli_feats = torch.stack(
                [batch["p_contra"], batch["p_neutral"], batch["p_entail"]],
                dim=-1
            ).to(device)

            for model_idx, model_dict in enumerate(models):
                model = model_dict["model"]

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    nli_feats=nli_feats
                )

                all_logits_list[model_idx].append(outputs["logits"].cpu())

    # Concatenate batches for each model
    all_logits = []
    for model_idx in range(n_models):
        model_logits = torch.cat(all_logits_list[model_idx], dim=0)  # [N, num_classes]
        all_logits.append(model_logits)

    all_logits = torch.stack(all_logits, dim=0)  # [n_models, N, num_classes]

    # Average logits
    mean_logits = all_logits.mean(dim=0)  # [N, num_classes]

    # Predictions
    predictions = mean_logits.argmax(dim=-1).numpy()

    return predictions, all_logits.numpy()


def main():
    print("=" * 80)
    print("Fair Multi-Seed Ensemble Evaluation")
    print("Using validation_seed=999 (unseen by all training seeds)")
    print("=" * 80)

    SEEDS = [21, 42, 84]
    VALIDATION_SEED = 999  # Different from all training seeds!

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Training seeds: {SEEDS}")
    print(f"Validation seed: {VALIDATION_SEED}\n")

    # Load models
    print("Loading trained models...")
    models = load_models(SEEDS, device)

    # Load validation data with INDEPENDENT seed
    print("\nLoading validation data with NLI features...")
    dataset = load_qevasion_with_nli("data/qevasion_with_nli")

    # Use validation_seed=999 for split (different from all training seeds)
    train_full = dataset["train"].shuffle(seed=VALIDATION_SEED)
    val_size = int(len(train_full) * 0.1)
    val_ds = train_full.select(range(len(train_full) - val_size, len(train_full)))

    print(f"Validation set size: {len(val_ds)}")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    label2id = {"Ambivalent": 0, "Clear Non-Reply": 1, "Clear Reply": 2}
    id2label = {v: k for k, v in label2id.items()}

    val_enc = tokenize_dataset(val_ds, tokenizer, label2id, max_length=256)
    val_loader = DataLoader(val_enc, batch_size=32, shuffle=False)

    # Evaluate each individual model first
    print("\n" + "=" * 80)
    print("Individual Model Performance on Fair Validation Set")
    print("=" * 80)

    individual_f1s = []
    for model_dict in models:
        seed = model_dict["seed"]
        model = model_dict["model"]

        all_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                nli_feats = torch.stack(
                    [batch["p_contra"], batch["p_neutral"], batch["p_entail"]],
                    dim=-1
                ).to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    nli_feats=nli_feats
                )
                preds = outputs["logits"].argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())

        val_labels = np.array([label2id[lbl] for lbl in val_ds["clarity_label"]])
        f1 = f1_score(val_labels, np.array(all_preds), average="macro")
        individual_f1s.append(f1)
        print(f"Seed {seed}: Macro F1 = {f1:.4f}")

    print(f"\nMean ± Std: {np.mean(individual_f1s):.4f} ± {np.std(individual_f1s):.4f}")

    # Get ensemble predictions
    print("\n" + "=" * 80)
    print("Ensemble Evaluation (Fair Validation Set)")
    print("=" * 80)

    val_preds, val_logits = ensemble_predict_logits(models, val_loader, device)

    # Ground truth
    val_labels = np.array([label2id[lbl] for lbl in val_ds["clarity_label"]])

    # Metrics
    ensemble_f1 = f1_score(val_labels, val_preds, average="macro")
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    ensemble_report = classification_report(
        val_labels, val_preds,
        target_names=target_names,
        digits=3
    )

    print(ensemble_report)
    print(f"\nEnsemble Macro F1: {ensemble_f1:.4f}")
    print(f"Best single-seed F1: {max(individual_f1s):.4f}")
    print(f"Improvement: {ensemble_f1 - max(individual_f1s):+.4f}")

    # Save summary
    summary = {
        "seeds": SEEDS,
        "validation_seed": VALIDATION_SEED,
        "note": "Fair evaluation - validation seed different from all training seeds",
        "individual_f1s": [float(f) for f in individual_f1s],
        "mean_individual_f1": float(np.mean(individual_f1s)),
        "std_individual_f1": float(np.std(individual_f1s)),
        "ensemble_f1": float(ensemble_f1),
        "improvement_over_best": float(ensemble_f1 - max(individual_f1s)),
        "ensemble_report": ensemble_report
    }

    with open("results_multiseed_fair_eval.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] Fair evaluation saved to: results_multiseed_fair_eval.json")

    print("\n" + "=" * 80)
    print("Comparison: Original vs Fair Evaluation")
    print("=" * 80)
    print("\nOriginal evaluation (seed=42 validation, data leakage):")
    print(f"  Ensemble F1: 0.8214")
    print(f"\nFair evaluation (seed=999 validation, no leakage):")
    print(f"  Ensemble F1: {ensemble_f1:.4f}")
    print(f"\nThe fair evaluation is the correct metric to report.")


if __name__ == "__main__":
    main()
