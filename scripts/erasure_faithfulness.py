from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_num}: {e}") from e
    return rows


def erase_tokens(text: str, tokens: List[str], mask_token: str = "[MASK]") -> str:
    """
    Replace each attributed token with mask_token using word-boundary matching.
    Handles multi-character tokens and is case-insensitive.
    """
    erased = text
    for token in tokens:
        if not token or len(token) < 2:
            continue
        pattern = r"\b" + re.escape(token) + r"\b"
        erased = re.sub(pattern, mask_token, erased, flags=re.IGNORECASE)
    return erased


def get_class_prob(
    model,
    tokenizer,
    text: str,
    class_idx: int,
    max_length: int,
    device: str,
) -> float:
    """Return the softmax probability of class_idx for the given text."""
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = F.softmax(logits, dim=-1)
    return float(probs[0, class_idx].item())


def validate_rows(rows: List[Dict[str, Any]], required: set[str]) -> None:
    """
    Validate all rows before running the experiment so that malformed records
    fail early with a clear message rather than crashing mid-run.
    """
    for i, row in enumerate(rows):
        missing = required - set(row.keys())
        if missing:
            raise ValueError(
                f"Row {i} missing fields: {sorted(missing)}. "
                "Make sure you are using attribution JSONL output, "
                "not the raw prediction JSONL."
            )

        if not isinstance(row.get("top_tokens"), list):
            raise ValueError(
                f"Row {i} has invalid 'top_tokens' type: {type(row.get('top_tokens')).__name__}. "
                "Expected a list of token dictionaries."
            )

        if row.get("text") is None:
            raise ValueError(f"Row {i} has null text.")


# ── main function ─────────────────────────────────────────────────────────────

def run_erasure_faithfulness(
    input_jsonl: str,
    model_path: str,
    output_csv: str,
    top_k_erase: int = 5,
    max_length: int = 256,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the erasure faithfulness test on all examples in input_jsonl.

    Returns a DataFrame with one row per example, plus summary CSVs.
    """
    # ── setup ──────────────────────────────────────────────────────────────
    device = device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[INFO] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    id2label = getattr(model.config, "id2label", {})

    # ── load attribution results ───────────────────────────────────────────
    rows = load_jsonl(input_jsonl)
    print(f"[INFO] Loaded {len(rows)} examples from {input_jsonl}")

    if not rows:
        raise ValueError(f"No rows found in input JSONL: {input_jsonl}")

    # ── validate required fields across the full file ──────────────────────
    required = {"id", "text", "top_tokens", "pred_class_idx_fresh", "pred_prob_fresh"}
    validate_rows(rows, required)

    # ── run erasure ────────────────────────────────────────────────────────
    results = []

    for i, row in enumerate(rows, start=1):
        original_text = row["text"]
        top_tokens = row.get("top_tokens", [])
        pred_class_idx = int(row["pred_class_idx_fresh"])
        pred_prob_original = float(row["pred_prob_fresh"])
        profession = row.get("label_pred_stored", row.get("label_pred", None))
        gender = row.get("gender", None)
        example_id = row.get("id", i)

        # Take top-K tokens for erasure
        tokens_to_erase = [
            t["token"] for t in top_tokens[:top_k_erase]
            if isinstance(t, dict) and t.get("token")
        ]

        # Erase tokens from text
        erased_text = erase_tokens(original_text, tokens_to_erase)

        # Get prediction probability on erased text for the SAME class
        pred_prob_erased = get_class_prob(
            model, tokenizer, erased_text, pred_class_idx, max_length, device
        )

        # Faithfulness = probability drop after erasure
        faithfulness = pred_prob_original - pred_prob_erased

        results.append({
            "id": example_id,
            "profession": profession,
            "gender": gender,
            "pred_class_idx": pred_class_idx,
            "pred_label": id2label.get(pred_class_idx, str(pred_class_idx)),
            "pred_prob_original": round(pred_prob_original, 4),
            "pred_prob_erased": round(pred_prob_erased, 4),
            "faithfulness": round(faithfulness, 4),
            "tokens_erased": ", ".join(tokens_to_erase),
            "n_tokens_erased": len(tokens_to_erase),
        })

        if i % 100 == 0 or i == len(rows):
            mean_f = sum(r["faithfulness"] for r in results) / len(results)
            print(f"[INFO] {i}/{len(rows)} processed | mean faithfulness so far: {mean_f:.4f}")

    # ── save per-example results ───────────────────────────────────────────
    df = pd.DataFrame(results)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[INFO] Saved per-example results: {output_path}")

    # ── profession-level summary ───────────────────────────────────────────
    prof_summary = (
        df.groupby("profession", as_index=False)
        .agg(
            mean_faithfulness=("faithfulness", "mean"),
            median_faithfulness=("faithfulness", "median"),
            mean_prob_original=("pred_prob_original", "mean"),
            mean_prob_erased=("pred_prob_erased", "mean"),
            n=("faithfulness", "count"),
        )
        .sort_values("mean_faithfulness", ascending=False)
        .round(4)
    )

    prof_path = output_path.parent / (output_path.stem + "_by_profession.csv")
    prof_summary.to_csv(prof_path, index=False)
    print(f"[INFO] Saved profession summary:    {prof_path}")

    # ── profession + gender summary ────────────────────────────────────────
    if df["gender"].notna().any():
        gender_summary = (
            df.groupby(["profession", "gender"], as_index=False)
            .agg(
                mean_faithfulness=("faithfulness", "mean"),
                mean_prob_original=("pred_prob_original", "mean"),
                mean_prob_erased=("pred_prob_erased", "mean"),
                n=("faithfulness", "count"),
            )
            .sort_values(["profession", "gender"])
            .round(4)
        )
        gender_path = output_path.parent / (output_path.stem + "_by_profession_gender.csv")
        gender_summary.to_csv(gender_path, index=False)
        print(f"[INFO] Saved gender summary:        {gender_path}")

    # ── print quick overview ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FAITHFULNESS SUMMARY")
    print("=" * 60)
    print(f"  Examples processed : {len(df)}")
    print(f"  Tokens erased/ex   : {df['n_tokens_erased'].mean():.1f} (avg)")
    print(f"  Mean P(y|original) : {df['pred_prob_original'].mean():.4f}")
    print(f"  Mean P(y|erased)   : {df['pred_prob_erased'].mean():.4f}")
    print(f"  Mean faithfulness  : {df['faithfulness'].mean():.4f}")
    print(f"  Median faithfulness: {df['faithfulness'].median():.4f}")
    print(f"  % positive (>0)    : {(df['faithfulness'] > 0).mean() * 100:.1f}%")
    print()
    print("Top 5 most faithful professions:")
    print(prof_summary[["profession", "mean_faithfulness", "n"]].head(5).to_string(index=False))
    print()
    print("Top 5 least faithful professions:")
    print(prof_summary[["profession", "mean_faithfulness", "n"]].tail(5).to_string(index=False))

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Erasure faithfulness test for proxy token attributions."
    )
    parser.add_argument(
        "--input_jsonl", required=True,
        help="Attribution JSONL file (output of run_attribution_encoder.py)"
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Path to fine-tuned encoder checkpoint (same one used for attribution)"
    )
    parser.add_argument(
        "--output_csv", required=True,
        help="Output path for per-example faithfulness results"
    )
    parser.add_argument(
        "--top_k_erase", type=int, default=5,
        help="How many top-attributed tokens to erase (default: 5)"
    )
    parser.add_argument(
        "--max_length", type=int, default=256,
        help="Maximum sequence length for re-prediction (default: 256)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: cpu / cuda / mps (auto-detected if not specified)"
    )
    args = parser.parse_args()

    run_erasure_faithfulness(
        input_jsonl=args.input_jsonl,
        model_path=args.model_path,
        output_csv=args.output_csv,
        top_k_erase=args.top_k_erase,
        max_length=args.max_length,
        device=args.device,
    )


if __name__ == "__main__":
    main()