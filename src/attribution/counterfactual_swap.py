from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ── pronoun swap tables ────────────────────────────────────────────────────────
#
# Swap is directional:
#   M→F applied when original biography gender is M
#   F→M applied when original biography gender is F
#
# Known ambiguity: 'her' serves both as object pronoun ("I saw her") and
# possessive determiner ("her book").  We map both uniformly to 'his',
# following standard practice in the counterfactual data augmentation
# literature.
# Note: \bmr\b already matches "Mr" in "Mr." (word boundary exists before ".")
# so the dotted variants mr./mrs./ms. are redundant and omitted.

_MALE_TO_FEMALE: List[Tuple[str, str]] = [
    (r"\bhe\b",      "she"),
    (r"\bhim\b",     "her"),
    (r"\bhis\b",     "her"),
    (r"\bhimself\b", "herself"),
    (r"\bmr\b",      "ms"),
]

_FEMALE_TO_MALE: List[Tuple[str, str]] = [
    (r"\bshe\b",     "he"),
    (r"\bher\b",     "his"),   # uniform mapping; see note above
    (r"\bhers\b",    "his"),
    (r"\bherself\b", "himself"),
    (r"\bmrs\b",     "mr"),
    (r"\bms\b",      "mr"),
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _apply_case(original: str, replacement: str) -> str:
    """Apply the capitalisation pattern of *original* to *replacement*."""
    if original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _replace_preserving_case(
    text: str, pattern: str, replacement: str
) -> Tuple[str, int]:
    """
    Replace all case-insensitive matches of *pattern* with *replacement*,
    preserving the capitalisation of each individual match.

    Returns ``(new_text, n_replacements)``.
    """
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
    if not matches:
        return text, 0
    parts: List[str] = []
    prev_end = 0
    for m in matches:
        parts.append(text[prev_end : m.start()])
        parts.append(_apply_case(m.group(), replacement))
        prev_end = m.end()
    parts.append(text[prev_end:])
    return "".join(parts), len(matches)


def swap_pronouns(text: str, original_gender: str) -> Tuple[str, int]:
    """
    Apply a counterfactual pronoun swap to *text* based on *original_gender*.

    Parameters
    ----------
    text            : biography string
    original_gender : ``'M'`` to swap male→female; ``'F'`` to swap female→male

    Returns
    -------
    (swapped_text, n_replacements)
        *n_replacements* is 0 if gender is unknown or no pronouns were found.
    """
    if original_gender == "M":
        pairs = _MALE_TO_FEMALE
    elif original_gender == "F":
        pairs = _FEMALE_TO_MALE
    else:
        return text, 0

    total = 0
    result = text
    for pattern, replacement in pairs:
        result, n = _replace_preserving_case(result, pattern, replacement)
        total += n
    return result, total


def _build_label2id(model_config) -> Dict[str, int]:
    """Build a label→id mapping from model config, handling int/str key variants."""
    id2label: Dict = getattr(model_config, "id2label", {})
    return {v: int(k) for k, v in id2label.items()}


def _predict_full(
    model,
    tokenizer,
    text: str,
    max_length: int,
    device: str,
    id2label: Dict,
) -> Tuple[str, int, torch.Tensor]:
    """
    Run inference on *text*.

    Returns ``(predicted_label, predicted_class_idx, full_prob_tensor)``.
    """
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
    probs = F.softmax(logits, dim=-1).squeeze(0)  # shape: (num_labels,)
    idx = int(torch.argmax(probs).item())
    label = id2label.get(idx, id2label.get(str(idx), str(idx)))
    return label, idx, probs


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Bad JSON on line {line_num} of {path}: {e}"
                ) from e
    return rows


# ── main experiment ───────────────────────────────────────────────────────────

def run_counterfactual_swap(
    input_jsonl: str | Path,
    model_path: str | Path,
    output_csv: str | Path,
    max_length: int = 256,
    limit: Optional[int] = None,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Counterfactual pronoun swap experiment.

    For each biography the gender-matched pronouns are swapped (M→F or F→M),
    the model is re-run on the swapped text, and we record whether the
    occupation prediction flips.

    The model is also re-run on the *original* (unswapped) text so that:
    - flip comparisons use fresh predictions from this checkpoint rather than
      stored labels that may have been produced by a different model version;
    - a checkpoint-consistency check is reported (% stored label == fresh label).

    Parameters
    ----------
    input_jsonl : prediction JSONL (fields: id, text, label_pred, gender).
                  Must come from the same checkpoint supplied to model_path.
    model_path  : fine-tuned encoder checkpoint directory
    output_csv  : path for the per-example results CSV
    max_length  : tokenisation truncation length
    limit       : process only the first N rows (None = all)
    device      : 'cpu' / 'cuda' / 'mps' (auto-detected if None)

    Returns
    -------
    DataFrame with one row per processed example.
    Three CSVs are written:
      - <output_csv>                          per-example results
      - <stem>_by_profession.csv              profession-level summary
      - <stem>_by_profession_gender.csv       profession × gender summary
    """
    device = device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model:  {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    id2label: Dict = getattr(model.config, "id2label", {})
    label2id: Dict[str, int] = _build_label2id(model.config)

    rows = load_jsonl(input_jsonl)
    if limit is not None:
        rows = rows[:limit]
    print(f"[INFO] Loaded {len(rows)} examples")

    # ── validate ──────────────────────────────────────────────────────────
    for i, row in enumerate(rows):
        for field in ("id", "text", "label_pred", "gender"):
            if field not in row:
                raise ValueError(
                    f"Row {i} missing required field '{field}'. "
                    "Ensure you are using the prediction JSONL, "
                    "not the attribution JSONL."
                )
        if row["text"] is None:
            raise ValueError(f"Row {i} has null 'text'.")

    # ── run swap ──────────────────────────────────────────────────────────
    results: List[Dict[str, Any]] = []
    n_skipped_gender = 0
    n_skipped_no_pronouns = 0
    n_stored_fresh_disagree = 0
    n_label_not_in_config = 0

    for i, row in enumerate(rows, start=1):
        gender = row.get("gender")
        label_stored = row["label_pred"]
        text = row["text"]

        # Skip examples with unknown gender
        if gender not in ("M", "F"):
            n_skipped_gender += 1
            continue

        # Apply pronoun swap
        swapped_text, n_swapped = swap_pronouns(text, gender)

        # Skip examples where no pronouns were found (swap is a no-op)
        if n_swapped == 0:
            n_skipped_no_pronouns += 1
            continue

        # Re-predict on original text with this checkpoint (checkpoint sanity check)
        label_fresh, fresh_class_idx, probs_fresh = _predict_full(
            model, tokenizer, text, max_length, device, id2label
        )

        # Re-predict on swapped text
        label_swapped, _, probs_swapped = _predict_full(
            model, tokenizer, swapped_text, max_length, device, id2label
        )

        # Checkpoint consistency: stored label vs fresh prediction
        stored_fresh_agree = (label_stored == label_fresh)
        if not stored_fresh_agree:
            n_stored_fresh_disagree += 1

        # Log stored labels absent from the config vocabulary
        if label2id.get(label_stored) is None:
            n_label_not_in_config += 1

        # Probabilities for the fresh-original predicted class
        prob_original = float(probs_fresh[fresh_class_idx].item())
        prob_after_swap = float(probs_swapped[fresh_class_idx].item())

        results.append({
            "id": row["id"],
            "profession": label_fresh,
            "profession_stored": label_stored,
            "gender": gender,
            "swap_direction": "M→F" if gender == "M" else "F→M",
            "n_tokens_swapped": n_swapped,
            "pred_label_stored": label_stored,
            "pred_label_original_fresh": label_fresh,
            "stored_fresh_agree": stored_fresh_agree,
            "pred_label_swapped": label_swapped,
            "pred_flipped": label_fresh != label_swapped,
            "pred_prob_original_class_original": round(prob_original, 4),
            "pred_prob_original_class_after_swap": round(prob_after_swap, 4),
            "pred_prob_swapped_top": round(float(probs_swapped.max().item()), 4),
        })

        if i % 200 == 0 or i == len(rows):
            n_done = len(results)
            flip_rate = sum(r["pred_flipped"] for r in results) / max(n_done, 1)
            print(
                f"[INFO] {i}/{len(rows)} processed | "
                f"results so far: {n_done} | flip rate: {flip_rate:.3f}"
            )

    print(f"[INFO] Skipped {n_skipped_gender} rows (unknown gender)")
    print(f"[INFO] Skipped {n_skipped_no_pronouns} rows (no pronouns found)")
    if n_label_not_in_config > 0:
        print(
            f"[WARN] {n_label_not_in_config} rows had stored label_pred not found "
            "in model config label vocabulary — check JSONL/checkpoint consistency."
        )

    df = pd.DataFrame(results)

    if df.empty:
        print("[WARN] No results produced. Check that the JSONL contains "
              "gender labels ('M'/'F') and gendered pronouns in the text.")
        return df

    # ── add prob_drop column for aggregation ──────────────────────────────
    df["prob_drop"] = (
        df["pred_prob_original_class_original"] - df["pred_prob_original_class_after_swap"]
    )

    # ── save per-example CSV ───────────────────────────────────────────────
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n[INFO] Saved per-example results: {output_path}")

    # ── profession-level summary ───────────────────────────────────────────
    prof_summary = (
        df.groupby("profession", as_index=False)
        .agg(
            n=("pred_flipped", "count"),
            flip_rate=("pred_flipped", "mean"),
            mean_n_swapped=("n_tokens_swapped", "mean"),
            mean_prob_original_class_after_swap=("pred_prob_original_class_after_swap", "mean"),
            mean_prob_drop=("prob_drop", "mean"),
        )
        .sort_values("flip_rate", ascending=False)
        .round(4)
    )
    prof_path = output_path.parent / (output_path.stem + "_by_profession.csv")
    prof_summary.to_csv(prof_path, index=False)
    print(f"[INFO] Saved profession summary:         {prof_path}")

    # ── profession × gender summary ────────────────────────────────────────
    pg_summary = (
        df.groupby(["profession", "gender"], as_index=False)
        .agg(
            n=("pred_flipped", "count"),
            flip_rate=("pred_flipped", "mean"),
            mean_n_swapped=("n_tokens_swapped", "mean"),
            mean_prob_original_class_after_swap=("pred_prob_original_class_after_swap", "mean"),
            mean_prob_drop=("prob_drop", "mean"),
        )
        .sort_values(["profession", "gender"])
        .round(4)
    )
    pg_path = output_path.parent / (output_path.stem + "_by_profession_gender.csv")
    pg_summary.to_csv(pg_path, index=False)
    print(f"[INFO] Saved profession×gender summary:  {pg_path}")

    # ── print console summary ──────────────────────────────────────────────
    overall_flip = df["pred_flipped"].mean()
    stored_fresh_agree_rate = df["stored_fresh_agree"].mean()
    m_df = df[df["gender"] == "M"]
    f_df = df[df["gender"] == "F"]
    m_flip = m_df["pred_flipped"].mean() if not m_df.empty else float("nan")
    f_flip = f_df["pred_flipped"].mean() if not f_df.empty else float("nan")

    print("\n" + "=" * 60)
    print("COUNTERFACTUAL SWAP SUMMARY")
    print("=" * 60)
    print(f"  Examples processed      : {len(df)}")
    print(f"  Skipped (no gender)     : {n_skipped_gender}")
    print(f"  Skipped (no pronouns)   : {n_skipped_no_pronouns}")
    print(f"  Stored==fresh agree     : {stored_fresh_agree_rate:.1%}  "
          f"({n_stored_fresh_disagree} disagreements)")
    print(f"  Overall flip rate       : {overall_flip:.3f}")
    print(f"  Flip rate  M→F bios     : {m_flip:.3f}")
    print(f"  Flip rate  F→M bios     : {f_flip:.3f}")
    print(f"  Mean tokens swapped/ex  : {df['n_tokens_swapped'].mean():.1f}")
    print(f"  Mean prob drop (orig→swap): {df['prob_drop'].mean():.4f}")
    print()
    print("Most flip-prone professions:")
    print(prof_summary[["profession", "flip_rate", "n"]].head(5).to_string(index=False))
    print()
    print("Least flip-prone professions:")
    print(prof_summary[["profession", "flip_rate", "n"]].tail(5).to_string(index=False))

    return df