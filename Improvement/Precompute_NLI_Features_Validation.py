# Precompute_NLI_Features_Validation.py

import torch
from pathlib import Path
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification


NLI_MODEL_NAME = "roberta-large-mnli"  # RoBERTa MNLI NLI model on HF
MAX_LENGTH = 256 #Revise to use dynamic padding via the map batching
BATCH_SIZE = 32


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NLI] Using device: {device}")

    #Load local CSV as a HF Dataset
    dataset = load_dataset(
        "csv",
        data_files={"test": "clarity_task_evaluation_dataset_cleaned.csv"}
    )
    
    print(f"[NLI] Loaded splits: {list(dataset.keys())}")
    print(f"[NLI] Loading NLI model: {NLI_MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(device)
    model.eval()

    use_fp16 = torch.cuda.is_available()
    if use_fp16:
        model = model.half()  # do this ONCE, outside add_nli

    def add_nli(batch):
        questions = batch["interview_question"]
        answers = batch["interview_answer"]

        enc = tokenizer(
            answers,        # premise
            questions,      # hypothesis
            truncation=True,
            padding=True,   # dynamic padding
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            if use_fp16:
                with torch.cuda.amp.autocast():
                    logits = model(**enc).logits
            else:
                logits = model(**enc).logits

            probs = torch.softmax(logits, dim=-1).float().cpu()

        return {
            "p_contra": probs[:, 0].tolist(),
            "p_neutral": probs[:, 1].tolist(),
            "p_entail": probs[:, 2].tolist(),
        }
  


    print(f"[NLI] Computing NLI features for split 'test' with {len(dataset['test'])} examples...")
    ds_with_nli = dataset["test"].map(add_nli, batched=True, batch_size=BATCH_SIZE)

    new_dataset = DatasetDict({"test": ds_with_nli})

    out_dir = Path("data/clarity_task_eval_cleaned_with_nli")
    out_dir.mkdir(parents=True, exist_ok=True)


    print(f"[NLI] Saving dataset with NLI features to '{out_dir}'...")
    new_dataset.save_to_disk(str(out_dir))
    print("[NLI] Done.")


if __name__ == "__main__":
    main()
