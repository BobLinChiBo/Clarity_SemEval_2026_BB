#Task_2_precompute_nli_features.py
"""
Compute NLI (entailment / neutral / contradiction), topic-shift probability,
and answer perplexity features for each (question, answer) pair in the
QEvasion dataset for Task 2 (9-category evasion classification).

Features added:
    - p_contra        : probability the answer contradicts the question
    - p_neutral       : probability the answer is unrelated
    - p_entail        : probability the answer entails the question
    - p_topic_shift   : probability that the answer shifts away from the topic
    - p_answer_ppl    : GPT-2 perplexity of the answer (fluency/informativeness)

Models used:
    - roberta-large-mnli for NLI inference
    - a fine-tuned topic shift detector (roberta-base)
    - GPT-2 for perplexity estimation (sequence truncated to 1024 tokens)

Output:
    data/qevasion_with_features
A HuggingFace Dataset containing the original fields plus 5 feature columns,
ready for downstream Task 2 training.

"""



import torch
from pathlib import Path

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    GPT2TokenizerFast,
    GPT2LMHeadModel
)

# -----------------------------
# MODELS USED
# -----------------------------
NLI_MODEL_NAME = "roberta-large-mnli"
TOPIC_MODEL_NAME = "roberta-base"
PPL_MODEL_NAME = "gpt2"

MAX_LENGTH = 256
BATCH_SIZE = 32


def compute_perplexity(texts, tokenizer, model, device):
    results = []
    for t in texts:
        enc = tokenizer(
            t,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
            ppl = torch.exp(out.loss).item()
        results.append(ppl)
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Features] Using device: {device}")

    # -----------------------------
    # Load QEvasion dataset
    # -----------------------------
    print("[Features] Loading QEvasion dataset...")
    dataset = load_dataset("ailsntua/QEvasion")

    # -----------------------------
    # Load NLI model
    # -----------------------------
    print(f"[Features] Loading NLI model: {NLI_MODEL_NAME}")
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(device)
    nli_model.eval()

    # -----------------------------
    # Load topic shift encoder (AutoModel for embeddings)
    # -----------------------------
    print(f"[Features] Loading topic shift encoder: {TOPIC_MODEL_NAME}")
    topic_tokenizer = AutoTokenizer.from_pretrained(TOPIC_MODEL_NAME)
    topic_model = AutoModel.from_pretrained(TOPIC_MODEL_NAME).to(device)
    topic_model.eval()

    # -----------------------------
    # Load Perplexity LM
    # -----------------------------
    print(f"[Features] Loading perplexity model: {PPL_MODEL_NAME}")
    ppl_tokenizer = GPT2TokenizerFast.from_pretrained(PPL_MODEL_NAME)
    ppl_model = GPT2LMHeadModel.from_pretrained(PPL_MODEL_NAME).to(device)
    ppl_model.eval()

    # ======================================================
    # Feature computation function
    # ======================================================
    def compute_features(batch):
        questions = batch["interview_question"]
        answers = batch["interview_answer"]

        # -------- NLI inference --------
        enc_nli = nli_tokenizer(
            answers, questions,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = nli_model(**enc_nli).logits
            probs = torch.softmax(logits, dim=-1).cpu()

        p_contra = probs[:, 0].tolist()
        p_neutral = probs[:, 1].tolist()
        p_entail = probs[:, 2].tolist()

        # -------- Topic shift --------
        enc_q = topic_tokenizer(
            questions,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(device)

        enc_a = topic_tokenizer(
            answers,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            q_emb = topic_model(**enc_q).last_hidden_state[:, 0]
            a_emb = topic_model(**enc_a).last_hidden_state[:, 0]

        sim = torch.nn.functional.cosine_similarity(q_emb, a_emb)
        p_topic_shift = (1 - sim).cpu().tolist()

        # -------- Perplexity --------
        perplexities = compute_perplexity(
            answers, ppl_tokenizer, ppl_model, device
        )

        return {
            "p_contra": p_contra,
            "p_neutral": p_neutral,
            "p_entail": p_entail,
            "p_topic_shift": p_topic_shift,
            "perplexity": perplexities,
        }

    # ======================================================
    # Apply feature computation to all splits
    # ======================================================
    new_splits = {}
    for split in dataset.keys():
        print(f"[Features] Computing features for split '{split}' ({len(dataset[split])} examples)...")

        ds_with_features = dataset[split].map(
            compute_features,
            batched=True,
            batch_size=BATCH_SIZE
        )
        new_splits[split] = ds_with_features

    final_dataset = DatasetDict(new_splits)

    # ======================================================
    # Save the enriched dataset
    # ======================================================
    out_dir = Path("data/qevasion_with_features")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Features] Saving enriched dataset to '{out_dir}'...")
    final_dataset.save_to_disk(str(out_dir))

    print("[Features] Done. QEvasion dataset now includes:")
    print("   - p_contra")
    print("   - p_neutral")
    print("   - p_entail")
    print("   - p_topic_shift")
    print("   - perplexity")


if __name__ == "__main__":
    main()
