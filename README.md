# CLARITY – SemEval 2026 Task 6, Subtask 1 (Clarity-Level Classification)

This repository implements a simple BERT-based baseline for **SemEval 2026 Task 6: CLARITY – Unmasking Political Question Evasions**, focusing on **Subtask 1: Clarity-level classification**. The system is implemented in `base.py` and uses the official **QEvasion** dataset released by the organizers.   

Subtask 1:  
Given a question and an answer from a political interview, predict whether the answer is:

- `Clear Reply`
- `Ambiguous` (called `Ambivalent` in the dataset)
- `Clear Non-Reply`   

Evaluation is based on **macro F1-score** across the three labels.   

---

## Repository structure

- `base.py`  
  End-to-end training and inference script for Subtask 1 (clarity-level classification).  
  - Loads the QEvasion dataset from Hugging Face.
  - Splits the official training set into train/validation (90% / 10%).
  - Fine-tunes BERT on the clarity labels.
  - Evaluates on the validation split.
  - Generates predictions for the official test split and writes them to `clarity_predictions.txt`.

You can extend this repository with additional scripts for Subtask 2 (evasion-level classification) or more advanced models.

---

## Task and data

### Task description

From the CLARITY shared task website:   

> Given a question and an answer, classify the answer as **Clear Reply**, **Ambiguous**, or **Clear Non-Reply** (Task 1).  
> Given a question and an answer, classify the answer into one of 9 evasion techniques (Task 2).

This repository currently targets **Task 1 only**.

### Dataset

We use the official QEvasion dataset hosted on Hugging Face: `ailsntua/QEvasion`.   

The dataset consists of question–answer pairs from presidential interviews, annotated with:

- `clarity_label` – one of:
  - `Clear Reply`
  - `Clear Non-Reply`
  - `Ambivalent` / `Ambivalent Reply` (this corresponds to the task label `Ambiguous`)
- `evasion_label` – one of 9 evasion techniques (for Task 2; not used in this script)
- Additional metadata such as `interview_question`, `interview_answer`, `president`, `date`, etc.

In `base.py`:

- We load the dataset via:

  ```python
  from datasets import load_dataset
  dataset = load_dataset("ailsntua/QEvasion")
  ```

* We use only:

  * `dataset["train"]` for training/validation.
  * `dataset["test"]` for generating predictions for submission.

---

## Method

The baseline model is:

* **Encoder**: `bert-base-uncased` (Hugging Face Transformers).

* **Task**: Single-label sequence classification with 3 labels.

* **Input**: The question and answer are concatenated and fed as a pair to BERT:

  ```python
  tokenizer(
      examples["interview_question"],
      examples["interview_answer"],
      padding="max_length",
      truncation=True,
      max_length=256,
  )
  ```

* **Label mapping**: The model builds `label2id` and `id2label` from the actual labels present in the training data:

  * Example (depending on the dataset version):

    ```python
    ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
    ```
  * Note: for the shared task, `Ambivalent` is equivalent to the label `Ambiguous` in the task description.

* **Train/validation split**:

  * Shuffle the official train split with a fixed seed.
  * Use 10% of it as validation.

* **Loss / optimization**:

  * Cross-entropy loss (built-in to `AutoModelForSequenceClassification`).
  * Optimizer: `AdamW` with `lr = 2e-5`.
  * Epochs: 3.
  * Batch sizes:

    * Train: 16
    * Validation and test: 32

* **Device**:

  * The script automatically selects GPU if available:

    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/BobLinChiBo/Clarity_SemEval_2026_BB.git
cd Clarity_SemEval_2026
```

### 2. Create and activate a virtual environment (recommended)

On Windows (PowerShell):

```bash
py -3.13 -m venv .venv_clarity
.\.venv_clarity\Scripts\activate
```

Or with another Python version, adjust `-3.13` accordingly.

### 3. Install dependencies

Minimal dependencies:

```bash
pip install torch           # choose CUDA or CPU build as appropriate
pip install transformers
pip install datasets
pip install scikit-learn
```

Optional (for faster dataset download if you use xet-backed repos):

```bash
pip install "huggingface_hub[hf_xet]"
```

---

## Usage

### Train the model and generate predictions

From the repository root:

```bash
python base.py
```

The script will:

1. Download the QEvasion dataset (first run only).
2. Fine-tune `bert-base-uncased` on the clarity labels.
3. Print validation metrics after each epoch.
4. Generate predictions for the test split.
5. Save predictions to:

   * `clarity_predictions.txt`

Each line of `clarity_predictions.txt` corresponds to one example in the QEvasion test split, in original order, and contains a single label string (e.g., `Clear Reply`).

These predictions can be adapted into the format required by the official Codabench submission interface for Task 1.

---

## Current baseline performance

With the default hyperparameters in `base.py` (3 epochs, `bert-base-uncased`, max length 256, train/val = 90/10) and using GPU, a typical run gives:

* Validation macro F1: **0.625**

Per-class metrics (example run):

| Class           | Precision | Recall | F1   | Support |
| --------------- | --------- | ------ | ---- | ------- |
| Ambivalent      | 0.69      | 0.79   | 0.74 | 198     |
| Clear Non-Reply | 0.76      | 0.53   | 0.63 | 30      |
| Clear Reply     | 0.56      | 0.47   | 0.51 | 116     |

Overall:

* Accuracy: 0.66
* Macro F1: 0.63
* Weighted F1: 0.65

These numbers are intended as a simple baseline; there is substantial room for improvement (better models, longer context, multi-task training with the evasion labels, etc.).

---

## Customization

You can modify training behavior by editing the arguments to `train_clarity_model` at the bottom of `base.py`:

```python
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
```

Common things to change:

* `model_name`
  Try other encoders, e.g., `roberta-base`, `microsoft/deberta-v3-base`, etc.
* `num_epochs`
  Increase for better performance (with early stopping).
* `learning_rate`
  Tune learning rate for different models.
* `max_length` in `tokenize_and_encode`
  Increase if answers/questions are regularly truncated.

---

## References

* CLARITY Shared Task Website: “CLARITY – Unmasking Political Question Evasions (SemEval 2026 Task 6)”
* QEvasion dataset on Hugging Face: `ailsntua/QEvasion`
* Thomas et al. (2024):
  `"I Never Said That": A dataset, taxonomy and baselines on response clarity classification` (EMNLP 2024).

Please see the above paper and the official task website for more details on the dataset construction, annotation process, and baseline systems.

