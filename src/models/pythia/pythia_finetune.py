"""
Part 3 — pythia_finetune.py
Supervised fine-tuning of Pythia (LoRA / QLoRA) for occupation classification.

Reads the pre-processed JSONL files produced by Part 1's export_dataset_jsonl.py:
    <data_dir>/train.jsonl
    <data_dir>/dev.jsonl

Each JSONL line schema (from Part 1):
    {"id": 0, "text": "...", "label": "professor", "gender": "M"}

Colab usage
-----------
# 1. Install dependencies
!pip install -q transformers==4.40.0 peft==0.10.0 accelerate bitsandbytes scikit-learn tqdm

# 2. Run (example for pythia-410m)
!python pythia_finetune.py \
    --model_size 410m \
    --data_dir ./processed \
    --output_dir ./checkpoints

Supported sizes: 70m | 160m | 410m | 1b | 1.4b | 2.8b
Add --use_4bit for QLoRA (recommended for 1.4b+ on Colab T4)
"""

import argparse
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from sklearn.metrics import accuracy_score, f1_score


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"  Loaded {len(records):,} records from {path}")
    return records


def build_label_vocab(train_records: List[dict]):
    """Build sorted label2id / id2label from the training set."""
    labels   = sorted({r["label"] for r in train_records if r.get("label")})
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BiosDataset(Dataset):
    """
    Tokenised dataset backed by a list of Part-1 JSONL records.
    Records whose label is absent from label2id are silently skipped.
    """

    def __init__(
        self,
        records: List[dict],
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 256,
    ):
        self.items: List[dict] = []
        skipped = 0

        for r in records:
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
            self.items.append({
                **enc,
                "labels": label2id[label],
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
# 3.  Model builder (LoRA / QLoRA)
# ─────────────────────────────────────────────────────────────────────────────

def build_lora_model(
    model_name: str,
    num_labels: int,
    label2id: Dict,
    id2label: Dict,
    use_4bit: bool = False,
):
    print(f"[model] Loading {model_name}  (num_labels={num_labels}) ...")

    load_kwargs: dict = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Pythia uses EOS as pad

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        **load_kwargs,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA targets Pythia's merged QKV projection in every attention layer
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        target_modules=["query_key_value"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Metrics callback
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)

    # ── Load pre-processed JSONL from Part 1 ──────────────────────────────────
    print("[data] Loading pre-processed JSONL ...")
    train_records = load_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    dev_records   = load_jsonl(os.path.join(args.data_dir, "dev.jsonl"))

    if args.max_train_samples:
        random.shuffle(train_records)
        train_records = train_records[: args.max_train_samples]
        print(f"  Capped to {len(train_records):,} training samples.")

    # ── Label vocabulary (derived from training labels) ───────────────────────
    label2id, id2label = build_label_vocab(train_records)
    num_labels = len(label2id)
    print(f"[data] {num_labels} labels: {list(label2id.keys())}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name = f"EleutherAI/pythia-{args.model_size}"
    use_4bit   = args.use_4bit and torch.cuda.is_available()
    model, tokenizer = build_lora_model(
        model_name, num_labels, label2id, id2label, use_4bit=use_4bit
    )

    # ── Tokenise ──────────────────────────────────────────────────────────────
    print("[data] Tokenising ...")
    train_ds = BiosDataset(train_records, tokenizer, label2id, args.max_length)
    dev_ds   = BiosDataset(dev_records,   tokenizer, label2id, args.max_length)
    print(f"  train={len(train_ds):,}  dev={len(dev_ds):,}")

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # ── Training arguments ────────────────────────────────────────────────────
    out_dir = os.path.join(args.output_dir, f"pythia-{args.model_size}")
    os.makedirs(out_dir, exist_ok=True)

    is_cuda = torch.cuda.is_available()
    fp16 = False
    bf16 = is_cuda and torch.cuda.is_bf16_supported()  # H100 natively supports BF16

    # H100: enable TF32 for matrix multiplications (~1.5x speedup vs FP32)
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,  # H100 80GB: use 128
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=fp16,
        bf16=bf16,                    # BF16: faster & more stable than FP16 on H100
        logging_steps=200,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=4,     # H100 nodes have many CPU cores
        dataloader_pin_memory=True,   # faster CPU->GPU transfer
        group_by_length=True,         # batch similar-length seqs -> less padding waste
        torch_compile=True,           # ~20-30% speedup via torch.compile on H100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\n[train] Fine-tuning pythia-{args.model_size} ...")
    trainer.train()

    # ── Save best checkpoint + label metadata ─────────────────────────────────
    best_dir = os.path.join(out_dir, "best")
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)

    meta = {
        "model_name": model_name,
        "model_size": args.model_size,
        "num_labels": num_labels,
        "label2id":   label2id,
        "id2label":   {str(k): v for k, v in id2label.items()},
    }
    with open(os.path.join(best_dir, "label_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[done] Best checkpoint saved → {best_dir}")

    dev_results = trainer.evaluate(dev_ds)
    print(f"[eval] Dev → accuracy={dev_results.get('eval_accuracy'):.4f}  "
          f"macro_f1={dev_results.get('eval_macro_f1'):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Part 3: Fine-tune Pythia (LoRA) using pre-processed JSONL from Part 1"
    )
    p.add_argument("--model_size",        type=str,   default="410m",
                   help="Pythia size: 70m | 160m | 410m | 1b | 1.4b | 2.8b")
    p.add_argument("--data_dir",          type=str,   required=True,
                   help="Folder with train.jsonl & dev.jsonl (Part 1 output)")
    p.add_argument("--output_dir",        type=str,   default="./checkpoints")
    p.add_argument("--num_epochs",        type=int,   default=3)
    p.add_argument("--train_batch_size",  type=int,   default=128,
                   help="Per-device batch size; H100 80GB handles 128 for 410m, 64 for 1.4b")
    p.add_argument("--eval_batch_size",   type=int,   default=128)
    p.add_argument("--grad_accum",        type=int,   default=1,
                   help="Gradient accumulation (keep at 1 on H100 — large batch already)")
    p.add_argument("--lr",                type=float, default=2e-4)
    p.add_argument("--max_length",        type=int,   default=256)
    p.add_argument("--max_train_samples", type=int,   default=None,
                   help="Cap training samples (quick testing)")
    p.add_argument("--use_4bit",          action="store_true",
                   help="QLoRA 4-bit quantisation (recommended for 1.4b+)")
    p.add_argument("--seed",              type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
