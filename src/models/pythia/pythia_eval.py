"""
Part 3 — pythia_eval.py
Evaluate a fine-tuned Pythia LoRA checkpoint on the test split and dump
predictions in the shared JSONL schema.

Reads:
    <data_dir>/test.jsonl          ← from Part 1
    <checkpoint_dir>/label_meta.json  ← saved by pythia_finetune.py

Output schema:
    {"id":1, "text":"...", "label_true":"physician", "label_pred":"nurse",
     "gender":"F", "model":"pythia-410m", "regime":"finetuned",
     "score":-0.47, "conf":0.62}

Colab usage
-----------
!python pythia_eval.py \
    --model_size 410m \
    --checkpoint_dir ./checkpoints/pythia-410m/best \
    --data_dir ./processed \
    --output_dir ./results
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel, PeftConfig
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records):,} records from {path}")
    return records


def load_label_meta(checkpoint_dir: str):
    """Read label_meta.json saved by pythia_finetune.py."""
    meta_path = os.path.join(checkpoint_dir, "label_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"label_meta.json not found in {checkpoint_dir}. "
            "Run pythia_finetune.py first."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    label2id = meta["label2id"]
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    print(f"[meta] {meta['num_labels']} labels loaded from {meta_path}")
    return label2id, id2label


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestBiosDataset(Dataset):
    """
    Tokenised test set that also stores metadata (id, text, gender, label_true)
    needed to build the JSONL prediction records.
    """

    def __init__(
        self,
        records: List[dict],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 256,
    ):
        self.meta:  List[dict] = []   # raw metadata per example
        self.items: List[dict] = []   # tokenised tensors + label

        skipped = 0
        for idx, r in enumerate(records):
            label = r.get("label", "")
            if label not in label2id:
                skipped += 1
                continue
            enc = tokenizer(
                r["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            self.items.append({**enc, "labels": label2id[label]})
            self.meta.append({
                "id":         r.get("id", idx),
                "text":       r.get("text", ""),
                "label_true": label,
                "gender":     r.get("gender", "UNK"),
            })

        if skipped:
            print(f"  [dataset] Skipped {skipped} records with unseen labels.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "input_ids":      torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels":         torch.tensor(item["labels"], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Load fine-tuned model
# ─────────────────────────────────────────────────────────────────────────────

def load_finetuned_model(checkpoint_dir: str, label2id: Dict, id2label: Dict,
                         device: torch.device):
    """Load LoRA adapter saved by pythia_finetune.py, merge weights."""
    print(f"[model] Loading LoRA checkpoint from {checkpoint_dir} ...")
    peft_config = PeftConfig.from_pretrained(checkpoint_dir)
    base_name   = peft_config.base_model_name_or_path
    print(f"  Base model: {base_name}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    num_labels = len(label2id)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        ignore_mismatched_sizes=True,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model = model.merge_and_unload()   # merge LoRA into base weights
    model.eval().to(device)
    print("[model] LoRA weights merged. Ready for inference.")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    dataset: TestBiosDataset,
    id2label: Dict[int, str],
    device: torch.device,
    batch_size: int,
    model_size: str,
) -> List[dict]:
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    loader   = DataLoader(dataset, batch_size=batch_size,
                          shuffle=False, collate_fn=collator)

    all_probs:  List[np.ndarray] = []
    all_preds:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            true_labels = batch.pop("labels").cpu().numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.float()
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
            preds  = np.argmax(probs, axis=-1)

            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(true_labels)

    all_probs  = np.concatenate(all_probs)   # (N, num_labels)
    all_preds  = np.concatenate(all_preds)   # (N,)
    all_labels = np.concatenate(all_labels)  # (N,)

    records = []
    for i, (pred_id, true_id, probs) in enumerate(
        zip(all_preds, all_labels, all_probs)
    ):
        meta     = dataset.meta[i]
        top_prob = float(probs[pred_id])
        score    = float(np.log(top_prob + 1e-12))   # log-probability as score

        records.append({
            "id":         meta["id"],
            "text":       meta["text"],
            "label_true": meta["label_true"],
            "label_pred": id2label[int(pred_id)],
            "gender":     meta["gender"],
            "model":      f"pythia-{model_size}",
            "regime":     "finetuned",
            "score":      round(score, 6),
            "conf":       round(top_prob, 6),
        })

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_and_print_metrics(records: List[dict], label_names: List[str]):
    y_true = [r["label_true"] for r in records]
    y_pred = [r["label_pred"] for r in records]

    acc      = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report   = classification_report(y_true, y_pred, labels=label_names,
                                     zero_division=0)

    # Per-gender accuracy
    from collections import defaultdict
    correct = defaultdict(int)
    total   = defaultdict(int)
    for r in records:
        g = r["gender"]
        total[g]   += 1
        correct[g] += int(r["label_true"] == r["label_pred"])
    gender_acc = {g: round(correct[g] / n, 4) for g, n in total.items() if n}

    print("\n" + "=" * 60)
    print(f"  Model   : pythia-{records[0]['model'].split('-')[1]}")
    print(f"  Regime  : finetuned")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Samples : {len(records):,}")
    print(f"  Gender accuracy: {gender_acc}")
    print("=" * 60)
    print(report)

    return {
        "model":           records[0]["model"],
        "regime":          "finetuned",
        "accuracy":        round(acc, 4),
        "macro_f1":        round(macro_f1, 4),
        "num_samples":     len(records),
        "gender_accuracy": gender_acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")

    # ── Label metadata ────────────────────────────────────────────────────────
    label2id, id2label = load_label_meta(args.checkpoint_dir)
    label_names        = [id2label[i] for i in range(len(id2label))]

    # ── Model ─────────────────────────────────────────────────────────────────
    model, tokenizer = load_finetuned_model(
        args.checkpoint_dir, label2id, id2label, device
    )

    # ── Test data (pre-processed JSONL from Part 1) ───────────────────────────
    print("[data] Loading test.jsonl ...")
    test_records = load_jsonl(os.path.join(args.data_dir, "test.jsonl"))
    test_ds      = TestBiosDataset(test_records, tokenizer, label2id, args.max_length)
    print(f"  Test examples (after label filtering): {len(test_ds):,}")

    # ── Inference ─────────────────────────────────────────────────────────────
    predictions = run_inference(
        model, tokenizer, test_ds, id2label, device,
        batch_size=args.batch_size,
        model_size=args.model_size,
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_and_print_metrics(predictions, label_names)

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    size = args.model_size

    preds_path = os.path.join(args.output_dir, f"preds_pythia_{size}_finetuned.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[save] Predictions  → {preds_path}")

    metrics_path = os.path.join(args.output_dir, f"metrics_pythia_{size}_finetuned.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[save] Metrics      → {metrics_path}")

    report_path = os.path.join(args.output_dir, f"clf_report_pythia_{size}_finetuned.txt")
    y_true = [r["label_true"] for r in predictions]
    y_pred = [r["label_pred"] for r in predictions]
    with open(report_path, "w") as f:
        f.write(f"model: pythia-{size} | regime: finetuned\n\n")
        f.write(classification_report(y_true, y_pred, labels=label_names, zero_division=0))
    print(f"[save] Clf report   → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Part 3: Evaluate fine-tuned Pythia on pre-processed test JSONL"
    )
    p.add_argument("--model_size",      type=str, required=True,
                   help="Pythia size string used for output filenames, e.g. 410m")
    p.add_argument("--checkpoint_dir",  type=str, required=True,
                   help="Path to the 'best/' dir saved by pythia_finetune.py")
    p.add_argument("--data_dir",        type=str, required=True,
                   help="Folder containing test.jsonl (Part 1 output)")
    p.add_argument("--output_dir",      type=str, default="./results")
    p.add_argument("--batch_size",      type=int, default=32)
    p.add_argument("--max_length",      type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
