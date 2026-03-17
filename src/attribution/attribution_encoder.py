from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.attribution.token_utils import top_k_positive_merged_tokens


def load_jsonl(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_num} of {path}: {e}") from e
    return pd.DataFrame(rows)


def save_jsonl(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@dataclass
class AttributionOutput:
    raw_tokens: List[str]
    raw_scores: List[float]
    pred_class_idx: int
    pred_label_name: str
    pred_prob: float
    top_tokens: List[Dict[str, Any]]


class EncoderAttributor:
    def __init__(self, model_path: str | Path, device: Optional[str] = None):
        self.model_path = str(model_path)
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        self.embedding_layer = self.model.get_input_embeddings()
        self.ig = IntegratedGradients(self._forward_with_embeds)

        self.id2label = getattr(self.model.config, "id2label", None)
        self.label2id = getattr(self.model.config, "label2id", None)
        self.num_labels = getattr(self.model.config, "num_labels", None)

    def _forward_with_embeds(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs.logits

    @torch.no_grad()
    def predict(self, text: str, max_length: int = 256) -> Dict[str, Any]:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        logits = self.model(**enc).logits
        probs = F.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs, dim=-1).item())
        pred_prob = float(probs[0, pred_idx].item())

        pred_label_name = str(pred_idx)
        if self.id2label is not None:
            # keys may be int or str depending on save format weirdness
            pred_label_name = self.id2label.get(pred_idx, self.id2label.get(str(pred_idx), str(pred_idx)))

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "pred_idx": pred_idx,
            "pred_prob": pred_prob,
            "pred_label_name": pred_label_name,
        }

    def attribute_text(
        self,
        text: str,
        max_length: int = 256,
        n_steps: int = 32,
        top_k: int = 5,
    ) -> AttributionOutput:
        pred = self.predict(text, max_length=max_length)

        input_ids = pred["input_ids"]
        attention_mask = pred["attention_mask"]
        pred_idx = pred["pred_idx"]
        pred_prob = pred["pred_prob"]
        pred_label_name = pred["pred_label_name"]

        input_embeds = self.embedding_layer(input_ids)
        baseline_embeds = torch.zeros_like(input_embeds)

        attributions = self.ig.attribute(
            inputs=input_embeds,
            baselines=baseline_embeds,
            target=pred_idx,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
        )

        token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().tolist()
        raw_tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
        top_tokens = top_k_positive_merged_tokens(raw_tokens, token_scores, k=top_k)

        return AttributionOutput(
            raw_tokens=raw_tokens,
            raw_scores=token_scores,
            pred_class_idx=pred_idx,
            pred_label_name=pred_label_name,
            pred_prob=pred_prob,
            top_tokens=top_tokens,
        )


def run_attribution_on_jsonl(
    input_jsonl: str | Path,
    model_path: str | Path,
    output_jsonl: str | Path,
    limit: int = 50,
    max_length: int = 256,
    n_steps: int = 32,
    top_k: int = 5,
    device: Optional[str] = None,
) -> pd.DataFrame:
    df = load_jsonl(input_jsonl)

    required_cols = ["id", "text", "label_pred"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input JSONL missing required columns: {missing}")

    if df["id"].duplicated().any():
        dupes = df.loc[df["id"].duplicated(), "id"].tolist()[:10]
        raise ValueError(f"Duplicate ids found in input JSONL. Examples: {dupes}")

    if df["text"].isna().any():
        n_missing = int(df["text"].isna().sum())
        raise ValueError(f"Input JSONL has {n_missing} rows with missing text")

    df = df.head(limit).copy()

    attributor = EncoderAttributor(model_path=model_path, device=device)

    print(f"[INFO] Device: {attributor.device}")
    print(f"[INFO] num_labels from checkpoint: {attributor.num_labels}")
    print(f"[INFO] Processing {len(df)} rows")

    rows_out: List[Dict[str, Any]] = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        attr = attributor.attribute_text(
            text=row.text,
            max_length=max_length,
            n_steps=n_steps,
            top_k=top_k,
        )

        out = {
            "id": row.id,
            "text": row.text,
            "label_true": getattr(row, "label_true", None),
            "label_pred_stored": row.label_pred,
            "label_pred_fresh": attr.pred_label_name,
            "gender": getattr(row, "gender", None),
            "model": getattr(row, "model", None),
            "regime": getattr(row, "regime", None),
            "conf_stored": getattr(row, "conf", None),
            "pred_class_idx_fresh": attr.pred_class_idx,
            "pred_prob_fresh": attr.pred_prob,
            "top_tokens": attr.top_tokens,
            "raw_tokens": attr.raw_tokens,
            "raw_scores": attr.raw_scores,
        }
        rows_out.append(out)

        if i <= 5:
            print("-" * 80)
            print(f"[DEBUG] Example {i}")
            print(f"id                : {row.id}")
            print(f"stored label_pred  : {row.label_pred}")
            print(f"fresh label_pred   : {attr.pred_label_name}")
            print(f"fresh pred_prob    : {attr.pred_prob:.4f}")
            print("top tokens         :",
                  [(x["token"], round(x["score"], 4)) for x in attr.top_tokens])

        if i % 10 == 0 or i == len(df):
            print(f"[INFO] Processed {i}/{len(df)}")

    save_jsonl(rows_out, output_jsonl)
    print(f"[INFO] Saved attribution output to: {output_jsonl}")

    return pd.DataFrame(rows_out)