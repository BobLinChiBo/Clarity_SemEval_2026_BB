#!/usr/bin/env python3
"""
Simple script to compare training results from different model variants.
Reads results_*.json files and displays comparison table.
"""

import json
from pathlib import Path
from typing import Dict, Any


def load_result(filepath: str) -> Dict[str, Any]:
    """Load a single result JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def extract_per_class_f1(report_str: str) -> Dict[str, float]:
    """Extract per-class F1 scores from classification report string."""
    lines = report_str.strip().split('\n')
    f1_scores = {}

    for line in lines:
        parts = line.split()
        if len(parts) >= 4 and parts[0] in ['Ambivalent', 'Clear']:
            # Handle "Clear Non-Reply" and "Clear Reply"
            if parts[0] == 'Clear' and len(parts) >= 5:
                label = ' '.join(parts[:2])
                try:
                    f1 = float(parts[3])
                    f1_scores[label] = f1
                except ValueError:
                    pass
            elif parts[0] == 'Ambivalent':
                try:
                    f1 = float(parts[2])
                    f1_scores['Ambivalent'] = f1
                except ValueError:
                    pass

    return f1_scores


def main():
    print("=" * 80)
    print("CLARITY Model Comparison")
    print("=" * 80)

    # Load results (order matters for display)
    results = {
        "Baseline (no features)": load_result("results_baseline.json"),
        "NLI only": load_result("results_nli.json"),
        "NLI (class-weighted)": load_result("results_nli_weighted.json"),
        "NLI + Topic": load_result("results_nli_topic.json"),
    }

    # Filter out missing results
    available = {k: v for k, v in results.items() if v is not None}

    if not available:
        print("\n⚠️  No result files found!")
        print("Expected files: results_baseline.json, results_nli.json, results_nli_topic.json")
        print("\nRun training first:")
        print('  python base.py')
        print('  python base_nli.py')
        print('  python base_nli_topic.py')
        return

    print(f"\nFound {len(available)} result(s)\n")

    # Header
    print(f"{'Model':<25} {'Best F1':<10} {'Final F1':<10} {'LR':<10} {'Epochs':<8}")
    print("-" * 75)

    # Summary table
    for name, data in available.items():
        print(f"{name:<25} "
              f"{data['best_val_f1']:<10.4f} "
              f"{data['final_val_f1']:<10.4f} "
              f"{data['learning_rate']:<10.2e} "
              f"{data['num_epochs']:<8}")

    # Per-class breakdown
    print("\n" + "=" * 80)
    print("Per-Class F1 Scores")
    print("=" * 80)

    for name, data in available.items():
        print(f"\n{name}:")
        if 'final_classification_report' in data:
            per_class = extract_per_class_f1(data['final_classification_report'])
            for label, f1 in sorted(per_class.items()):
                print(f"  {label:<20} {f1:.3f}")
        else:
            print("  (classification report not available)")

    # Training curves
    print("\n" + "=" * 80)
    print("Training Curves")
    print("=" * 80)

    for name, data in available.items():
        print(f"\n{name}:")
        if 'training_history' in data:
            print(f"  {'Epoch':<8} {'Train Loss':<12} {'Val F1':<10}")
            print("  " + "-" * 35)
            for entry in data['training_history']:
                print(f"  {entry['epoch']:<8} "
                      f"{entry['train_loss']:<12.4f} "
                      f"{entry['val_f1']:<10.4f}")
        else:
            print("  (training history not available)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
