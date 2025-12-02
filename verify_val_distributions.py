#!/usr/bin/env python3
"""
Verify validation set distributions across different seeds.

This checks if shuffle(999) val set is unusually easy or has biased label distribution.
"""

import json
import torch
import numpy as np
from collections import Counter
from base_nli import load_qevasion_with_nli, tokenize_dataset, RobertaWithNLI
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from transformers import AutoTokenizer


def check_label_distribution(val_seed, dataset):
    """Check label distribution for a given validation seed."""
    train_full = dataset["train"].shuffle(seed=val_seed)
    val_size = int(len(train_full) * 0.1)
    val_ds = train_full.select(range(len(train_full) - val_size, len(train_full)))

    labels = [ex["clarity_label"] for ex in val_ds]
    label_counts = Counter(labels)

    return label_counts, len(val_ds)


def evaluate_model_on_val_seed(model, dataset, val_seed, tokenizer, label2id, id2label, device):
    """Evaluate a model on a specific validation split."""
    train_full = dataset["train"].shuffle(seed=val_seed)
    val_size = int(len(train_full) * 0.1)
    val_ds = train_full.select(range(len(train_full) - val_size, len(train_full)))

    val_enc = tokenize_dataset(val_ds, tokenizer, label2id, max_length=256)
    val_loader = DataLoader(val_enc, batch_size=32, shuffle=False)

    model.eval()
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

    return f1


def main():
    print("=" * 80)
    print("Validation Set Distribution Verification")
    print("=" * 80)

    # Load dataset
    dataset = load_qevasion_with_nli("data/qevasion_with_nli")

    # Check label distributions
    print("\n" + "=" * 80)
    print("Label Distributions Across Different Validation Seeds")
    print("=" * 80)

    val_seeds = [21, 42, 84, 999, 100, 200, 300]

    print(f"\n{'Seed':<8} {'Ambivalent':<12} {'CNR':<12} {'CR':<12} {'Total':<8}")
    print("-" * 80)

    distributions = {}
    for val_seed in val_seeds:
        label_counts, total = check_label_distribution(val_seed, dataset)
        distributions[val_seed] = label_counts

        amb = label_counts.get("Ambivalent", 0)
        cnr = label_counts.get("Clear Non-Reply", 0)
        cr = label_counts.get("Clear Reply", 0)

        print(f"{val_seed:<8} {amb:<12} {cnr:<12} {cr:<12} {total:<8}")

    # Calculate statistics
    print("\n" + "=" * 80)
    print("Distribution Statistics")
    print("=" * 80)

    amb_counts = [d.get("Ambivalent", 0) for d in distributions.values()]
    cnr_counts = [d.get("Clear Non-Reply", 0) for d in distributions.values()]
    cr_counts = [d.get("Clear Reply", 0) for d in distributions.values()]

    print(f"\nAmbivalent: {np.mean(amb_counts):.1f} ± {np.std(amb_counts):.1f} (range: {min(amb_counts)}-{max(amb_counts)})")
    print(f"Clear Non-Reply: {np.mean(cnr_counts):.1f} ± {np.std(cnr_counts):.1f} (range: {min(cnr_counts)}-{max(cnr_counts)})")
    print(f"Clear Reply: {np.mean(cr_counts):.1f} ± {np.std(cr_counts):.1f} (range: {min(cr_counts)}-{max(cr_counts)})")

    # Load models
    print("\n" + "=" * 80)
    print("Loading Trained Models")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    label2id = {"Ambivalent": 0, "Clear Non-Reply": 1, "Clear Reply": 2}
    id2label = {v: k for k, v in label2id.items()}

    models = {}
    for seed in [21, 42, 84]:
        checkpoint_path = f"model_nli_weighted_seed{seed}_roberta.pt"
        print(f"Loading {checkpoint_path}...")

        model = RobertaWithNLI(model_name="roberta-base", num_labels=3)
        state_dict = torch.load(checkpoint_path, map_location=device)
        if "class_weights" in state_dict:
            del state_dict["class_weights"]
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        models[seed] = model

    # Evaluate each model on each validation seed
    print("\n" + "=" * 80)
    print("Model Performance Across Different Validation Seeds")
    print("=" * 80)

    results = {}
    for model_seed in [21, 42, 84]:
        results[model_seed] = {}
        print(f"\nModel trained with seed={model_seed}:")

        for val_seed in val_seeds:
            f1 = evaluate_model_on_val_seed(
                models[model_seed],
                dataset,
                val_seed,
                tokenizer,
                label2id,
                id2label,
                device
            )
            results[model_seed][val_seed] = f1
            print(f"  Val seed {val_seed:<4}: F1 = {f1:.4f}")

    # Analyze results
    print("\n" + "=" * 80)
    print("Analysis: Average F1 on Each Validation Seed")
    print("=" * 80)

    print(f"\n{'Val Seed':<10} {'Mean F1':<10} {'Std F1':<10} {'Min F1':<10} {'Max F1':<10}")
    print("-" * 80)

    for val_seed in val_seeds:
        f1s = [results[model_seed][val_seed] for model_seed in [21, 42, 84]]
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        min_f1 = min(f1s)
        max_f1 = max(f1s)

        marker = " *" if val_seed == 999 else ""
        print(f"{val_seed:<10} {mean_f1:<10.4f} {std_f1:<10.4f} {min_f1:<10.4f} {max_f1:<10.4f}{marker}")

    # Find easiest and hardest validation seeds
    avg_f1s = {val_seed: np.mean([results[ms][val_seed] for ms in [21, 42, 84]])
               for val_seed in val_seeds}

    easiest_seed = max(avg_f1s, key=avg_f1s.get)
    hardest_seed = min(avg_f1s, key=avg_f1s.get)

    print("\n" + "=" * 80)
    print("Conclusion")
    print("=" * 80)

    print(f"\nEasiest validation seed: {easiest_seed} (avg F1 = {avg_f1s[easiest_seed]:.4f})")
    print(f"Hardest validation seed: {hardest_seed} (avg F1 = {avg_f1s[hardest_seed]:.4f})")
    print(f"Difference: {avg_f1s[easiest_seed] - avg_f1s[hardest_seed]:.4f}")

    print(f"\nSeed 999 (used for fair evaluation):")
    print(f"  Average F1: {avg_f1s[999]:.4f}")
    print(f"  Rank: {sorted(avg_f1s.values(), reverse=True).index(avg_f1s[999]) + 1}/{len(val_seeds)}")

    if avg_f1s[999] == avg_f1s[easiest_seed]:
        print("\n[!] WARNING: Seed 999 is the EASIEST validation set!")
        print("    Fair evaluation F1 (0.8047) may be optimistic.")
        print("    Expected test F1: 0.70-0.75")
    elif avg_f1s[999] > np.median(list(avg_f1s.values())):
        print("\n[!] Seed 999 is easier than median validation set.")
        print("    Fair evaluation F1 (0.8047) may be slightly optimistic.")
        print("    Expected test F1: 0.73-0.78")
    else:
        print("\n[OK] Seed 999 is representative (close to median difficulty).")
        print("     Fair evaluation F1 (0.8047) is reliable.")
        print("     Expected test F1: 0.77-0.82")

    # Save results
    summary = {
        "label_distributions": {str(k): dict(v) for k, v in distributions.items()},
        "model_performance": {
            str(model_seed): {str(val_seed): float(f1)
                              for val_seed, f1 in val_results.items()}
            for model_seed, val_results in results.items()
        },
        "average_f1_by_val_seed": {str(k): float(v) for k, v in avg_f1s.items()},
        "easiest_seed": int(easiest_seed),
        "hardest_seed": int(hardest_seed),
        "seed_999_rank": sorted(avg_f1s.values(), reverse=True).index(avg_f1s[999]) + 1,
        "seed_999_f1": float(avg_f1s[999])
    }

    with open("validation_seed_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[OK] Analysis saved to: validation_seed_analysis.json")


if __name__ == "__main__":
    main()
