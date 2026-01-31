from torch.utils.data import DataLoader

def predict_external_and_save_txt(
    model,
    tokenizer,
    id2label: Dict[int, str],
    external_dir: str = "data/clarity_task_eval_cleaned_with_nli",
    max_length: int = 256,
    batch_size: int = 32,
    t_reply: float = 0.70,
    t_cnr: float = 0.70,
    output_path: str = "clarity_predictions_external.txt",
):
    """
    Apply trained hierarchical model to an EXTERNAL dataset saved_to_disk that contains:
      - interview_question, interview_answer
      - p_contra, p_neutral, p_entail
    and save predicted clarity labels (strings) to a txt file (one per line).
    """
    device = next(model.parameters()).device
    model.eval()

    # 1) load external dataset with NLI already computed
    ext = load_external_with_nli(external_dir)      # DatasetDict
    ext_test = ext["test"]                          # Dataset

    # 2) tokenize WITHOUT labels
    ext_enc = tokenize_dataset_no_labels(ext_test, tokenizer, max_length=max_length)

    # 3) dataloader
    ext_loader = DataLoader(ext_enc, batch_size=batch_size, shuffle=False)

    # 4) predict ids with thresholded decoding
    pred_ids = predict_thresholded(
        model=model,
        dataloader=ext_loader,
        device=device,
        id2label=id2label,
        t_reply=t_reply,
        t_cnr=t_cnr,
    )

    # 5) map ids -> label strings and save
    pred_labels = [id2label[i] for i in pred_ids]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for lbl in pred_labels:
            f.write(lbl + "\n")

    print(f"[OK] Saved {len(pred_labels)} predictions to: {out_path.resolve()}")

predict_external_and_save_txt(model=model,tokenizer=tokenizer,id2label=id2label,external_dir="data/clarity_task_eval_cleaned_with_nli",max_length=256,batch_size=32,t_reply=t_reply,t_cnr=t_cnr,output_path="clarity_predictions_external_hier.txt",)
