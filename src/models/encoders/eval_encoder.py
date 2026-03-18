import argparse
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../data"))
import numpy as np
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score

from src.data.data import load_bios

#The file loads a finetuned encoder checkpoint, runs it on the BIOS test split,
# prints Macro-F1/Accuracy, then writes per-example predictions to a JSONL file

#turns logits into probabilities that sum to 1 across classes, used for confidence
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


@torch.no_grad() #disable gradient tracking
def main():
    parser = argparse.ArgumentParser()
    #run command: python eval_encoder.py --model_dir checkpoints/roberta-base --model_tag roberta-ft --out_jsonl data/preds.jsonl

    #directory with model weights+config+tokenizer files
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path produced by train_encoder.py (checkpoints/roberta-base or checkpoints/distilbert-base-uncased)")
    #encoder
    parser.add_argument("--model_tag", type=str, required=True,
                        choices=["distilbert-ft", "roberta-ft"],
                        help="Used in JSONL 'model' field.")
    #out dir for predictions
    parser.add_argument("--out_jsonl", type=str, required=True)

    #masked or unmasked data pipeline
    parser.add_argument("--data_regime", type=str, default="unmasked",
                        choices=["masked", "unmasked"])
    parser.add_argument("--ids_jsonl", type=str, default="results/pythia/preds_pythia_1.4b_fewshot.jsonl",
                        help="only examples whose ids appear in this file are kept.")
    #max tokens per input
    parser.add_argument("--max_length", type=int, default=256)


    #batch size per cpu/gpu device
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    #load dataset + shared label mapping
    if args.data_regime == "masked":
        from src.data.data import BiosConfig
        cfg = BiosConfig(
            mask_gender=True,
            mask_titles=True,
            mask_gendered_nouns=True,
            mask_label_leakage=True,
        )
        ds, label2id, id2label, meta = load_bios(cfg=cfg)
    else:
        ds, label2id, id2label, meta = load_bios()
    test = ds["test"]

    label_list = [id2label[i] for i in range(len(id2label))]
    if any(str(label).isdigit() for label in label_list):
        raise ValueError("load_bios() returned numeric labels instead of profession names.")

    if args.ids_jsonl is not None:
        ordered_ids = []
        with open(args.ids_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                ordered_ids.append(int(row["id"]))

        selected_ids = set(ordered_ids)
        test = test.filter(lambda ex: int(ex["id"]) in selected_ids)

        #reorder the filtered test set to match the exact id order in the reference jsonl
        id_to_idx = {int(example_id): idx for idx, example_id in enumerate(test["id"])}
        test = test.select([id_to_idx[i] for i in ordered_ids if i in id_to_idx])

        #print(f"test set to {len(test)} rows using ids from: {args.ids_jsonl}")
    # with open("data/profession_mapping.json", "r", encoding="utf-8") as f:
    #     profession_mapping = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    model.eval()
    #run on gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #manually loop in batches and call model(**enc)
    ids = list(test["id"])
    texts = list(test["text"])
    gold_labels = list(test["label"])
    gold_label_ids = list(test["label_id"])
    genders = list(test["gender"])
    if args.ids_jsonl is not None and len(ids) != 3000:
        print(f"Warning: expected 3000 rows after filtering, got {len(ids)}")

    #convert gold string labels to numeric class ids
    gold_ids = np.array(gold_label_ids, dtype=int)

    #store logits for each batch, then concatenate at the end
    all_logits: List[np.ndarray] = []

    #loop through dataset in batches
    for i in range(0, len(texts), args.batch_size):
        #slice the list of texts to get the current batch
        batch_texts = texts[i:i + args.batch_size]

        #tokenize the batch and move tensors to the same device as the model
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=args.max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)

        #run the model forward pass using the encoded inputs
        out = model(**enc)

        #extract raw class scores (logits) for each example
        logits = out.logits.detach().cpu().numpy()

        #store them
        all_logits.append(logits)
    #add all logits into one array
    logits = np.concatenate(all_logits, axis=0)
    #convert logits into probabilities
    probs = softmax(logits, axis=-1)
    #predicted index
    pred_ids = np.argmax(probs, axis=-1)

    macro_f1 = f1_score(gold_ids, pred_ids, average="macro")
    acc = accuracy_score(gold_ids, pred_ids)
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Accuracy: {acc:.4f}")

    #create output folder if it does not exist
    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    #save results
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i in range(len(ids)):
            pred_label = id2label.get(int(pred_ids[i]), "UNK")
            conf = float(np.max(probs[i]))
            score = float(np.max(logits[i]))

            row = {
                "id": ids[i],
                "text": texts[i],
                "label_true": gold_labels[i],
                "label_pred": pred_label,
                "gender": genders[i],
                "model": args.model_tag,
                "regime": "finetuned",
                "score": score,
                "conf": conf
            }
            f.write(json.dumps(row) + "\n")

    print(f"Wrote predictions to: {args.out_jsonl}")


if __name__ == "__main__":
    main()