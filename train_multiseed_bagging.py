#!/usr/bin/env python3
"""
Multi-seed bagging for NLI class-weighted model.

Trains the best model (NLI + class weighting) with multiple seeds,
then ensembles by averaging logits to reduce variance.

This helps stabilize performance on small datasets (~3k training examples).
"""

import json
import torch
import numpy as np
from pathlib import Path
from base_nli import train_clarity_with_nli
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report


def train_multiple_seeds(seeds, num_epochs=10):
    """Train model with different seeds and save all checkpoints."""

    models = []
    results = {}

    for seed_idx, seed in enumerate(seeds, 1):
        print("\n" + "=" * 80)
        print(f"Training model {seed_idx}/{len(seeds)} with seed={seed}")
        print("=" * 80)

        model, tokenizer, label2id, id2label, test_loader = train_clarity_with_nli(
            model_name="roberta-base",
            data_dir="data/qevasion_with_nli",
            num_epochs=num_epochs,
            train_batch_size=16,
            eval_batch_size=32,
            learning_rate=2e-5,
            max_length=256,
            seed=seed,
            use_class_weights=True,
            model_nickname="roberta",
        )

        # Checkpoint is now saved by train_clarity_with_nli with unique name
        checkpoint_path = f"model_nli_weighted_seed{seed}_roberta.pt"
        print(f"Checkpoint saved by train_clarity_with_nli: {checkpoint_path}")

        models.append({
            "seed": seed,
            "model": model,
            "checkpoint": checkpoint_path
        })

        # Load training results with new naming scheme
        results_path = f"results_nli_weighted_seed{seed}_roberta.json"
        with open(results_path, "r") as f:
            result = json.load(f)

        results[f"seed_{seed}"] = {
            "best_val_f1": result["best_val_f1"],
            "final_val_f1": result["final_val_f1"]
        }

    # Return last tokenizer/label mappings (same for all seeds)
    return models, tokenizer, label2id, id2label, test_loader, results


def ensemble_predict_logits(models, data_loader, device, id2label):
    """
    Ensemble prediction by averaging logits across all models.

    Args:
        models: List of dicts with 'model' key
        data_loader: DataLoader for the dataset
        device: torch device

    Returns:
        predictions: [N] final predictions
        all_logits: [n_models, N, num_classes] logits from each model
    """

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
    print("Multi-Seed Bagging for NLI Class-Weighted Model")
    print("=" * 80)

    SEEDS = [21, 42, 84]
    NUM_EPOCHS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Seeds: {SEEDS}")
    print(f"Epochs per seed: {NUM_EPOCHS}\n")

    # Train all seeds
    models, tokenizer, label2id, id2label, test_loader, training_results = train_multiple_seeds(
        seeds=SEEDS,
        num_epochs=NUM_EPOCHS
    )

    print("\n" + "=" * 80)
    print("Training Summary Across Seeds")
    print("=" * 80 + "\n")

    val_f1s = []
    for seed_key, result in training_results.items():
        seed = seed_key.split("_")[1]
        best_f1 = result["best_val_f1"]
        val_f1s.append(best_f1)
        print(f"Seed {seed:>3}: Best Val F1 = {best_f1:.4f}")

    mean_f1 = np.mean(val_f1s)
    std_f1 = np.std(val_f1s)
    print(f"\nMean ± Std: {mean_f1:.4f} ± {std_f1:.4f}")

    # Ensemble validation predictions
    print("\n" + "=" * 80)
    print("Ensemble Evaluation (Validation Set)")
    print("=" * 80)

    from base_nli import load_qevasion_with_nli, tokenize_dataset

    dataset = load_qevasion_with_nli("data/qevasion_with_nli")
    train_full = dataset["train"].shuffle(seed=42)  # Use seed=42 for val split
    val_size = int(len(train_full) * 0.1)
    val_ds = train_full.select(range(len(train_full) - val_size, len(train_full)))

    val_enc = tokenize_dataset(val_ds, tokenizer, label2id, max_length=256)
    val_loader = DataLoader(val_enc, batch_size=32, shuffle=False)

    # Get ensemble predictions
    val_preds, val_logits = ensemble_predict_logits(models, val_loader, device, id2label)

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
    print(f"Best single-seed F1: {max(val_f1s):.4f}")
    print(f"Improvement: {ensemble_f1 - max(val_f1s):+.4f}")

    # Generate test predictions
    print("\n" + "=" * 80)
    print("Generating Test Predictions (Ensemble)")
    print("=" * 80)

    test_preds, test_logits = ensemble_predict_logits(models, test_loader, device, id2label)

    output_path = "clarity_predictions_multiseed_ensemble.txt"
    with open(output_path, "w") as f:
        for pred_id in test_preds:
            f.write(id2label[pred_id] + "\n")

    print(f"Saved test predictions to: {output_path}")

    # Save summary
    summary = {
        "seeds": SEEDS,
        "num_epochs": NUM_EPOCHS,
        "training_results": training_results,
        "ensemble_val_f1": float(ensemble_f1),
        "individual_val_f1s": [float(f) for f in val_f1s],
        "mean_val_f1": float(mean_f1),
        "std_val_f1": float(std_f1),
        "ensemble_val_report": ensemble_report
    }

    with open("results_multiseed_ensemble.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] Summary saved to: results_multiseed_ensemble.json")

    print("\n" + "=" * 80)
    print("Multi-Seed Bagging Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
