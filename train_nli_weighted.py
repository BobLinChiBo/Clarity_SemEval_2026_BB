#!/usr/bin/env python3
"""
Train NLI sub-network model with class-weighted loss.

This script trains the same architecture as base_nli.py but with
inverse frequency class weights to handle class imbalance.
"""

from base_nli import train_clarity_with_nli, predict_and_save

if __name__ == "__main__":
    print("=" * 80)
    print("Training NLI Sub-Network with Class-Weighted Loss")
    print("=" * 80)

    model, tokenizer, label2id, id2label, test_loader = train_clarity_with_nli(
        model_name="roberta-base",
        data_dir="data/qevasion_with_nli",
        num_epochs=10,  # Use 10 epochs to match best result
        train_batch_size=16,
        eval_batch_size=32,
        learning_rate=2e-5,
        max_length=256,
        seed=42,
        use_class_weights=True,  # Enable class weighting
        model_nickname="roberta",  # Explicit nickname
    )

    predict_and_save(
        model,
        test_loader,
        id2label,
        output_path="clarity_predictions_nli_weighted_seed42_roberta.txt"
    )

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("Results saved to: results_nli_weighted_seed42_roberta.json")
    print("Predictions saved to: clarity_predictions_nli_weighted_seed42_roberta.txt")
    print("=" * 80)
