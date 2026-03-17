from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Iterable

import pandas as pd


DEFAULT_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with",
    "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "he", "she", "his", "her", "him", "hers", "their", "they", "them",
    "this", "that", "these", "those", "who", "which", "where", "when",
    "has", "have", "had", "it", "its", "also", "after", "before", "into",
    "about", "over", "under", "between", "during"
}


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
                raise ValueError(f"Bad JSON on line {line_num} of {path}: {e}") from e
    return rows


def _normalize_token(token: str) -> str:
    token = token.strip().lower()

    # remove leading/trailing punctuation but keep internal underscores/hyphens
    token = re.sub(r"^[^\w]+|[^\w]+$", "", token)

    return token


def _is_bad_token(
    token: str,
    min_len: int = 3,
    stopwords: Optional[set[str]] = None,
) -> bool:
    if not token:
        return True
    if len(token) < min_len:
        return True
    if token.isdigit():
        return True
    if stopwords and token in stopwords:
        return True
    if all(not ch.isalnum() for ch in token):
        return True
    return False


def aggregate_proxy_tokens(
    attribution_rows: Iterable[Dict[str, Any]],
    group_by_gender: bool = False,
    min_token_len: int = 3,
    remove_stopwords: bool = True,
    min_count: int = 1,
    min_doc_freq: int = 1,
) -> pd.DataFrame:
    """
    Aggregate top_tokens by predicted profession, optionally split by gender.
    """
    stopwords = DEFAULT_STOPWORDS if remove_stopwords else set()

    agg = defaultdict(lambda: {
        "count_topk": 0,
        "score_sum": 0.0,
        "example_ids": set(),
    })

    n_rows = 0
    n_rows_missing_group = 0
    n_tokens_seen = 0
    n_tokens_kept = 0

    for row in attribution_rows:
        n_rows += 1

        profession = row.get("label_pred_stored")
        if profession is None:
            n_rows_missing_group += 1
            continue

        gender = row.get("gender", None)
        top_tokens = row.get("top_tokens", [])
        ex_id = row.get("id")

        if not isinstance(top_tokens, list):
            continue

        for item in top_tokens:
            if not isinstance(item, dict):
                continue

            token = _normalize_token(str(item.get("token", "")))
            score = float(item.get("score", 0.0))

            n_tokens_seen += 1

            if _is_bad_token(token, min_len=min_token_len, stopwords=stopwords):
                continue
            if score <= 0:
                continue

            n_tokens_kept += 1

            if group_by_gender:
                key = (profession, gender, token)
            else:
                key = (profession, token)

            agg[key]["count_topk"] += 1
            agg[key]["score_sum"] += score
            agg[key]["example_ids"].add(ex_id)

    rows_out = []
    for key, stats in agg.items():
        if group_by_gender:
            profession, gender, token = key
        else:
            profession, token = key
            gender = None

        count_topk = stats["count_topk"]
        doc_freq = len(stats["example_ids"])

        if count_topk < min_count:
            continue
        if doc_freq < min_doc_freq:
            continue

        mean_attr = stats["score_sum"] / count_topk
        weighted_score = count_topk * mean_attr

        rows_out.append({
            "profession": profession,
            "gender": gender,
            "token": token,
            "count_topk": count_topk,
            "doc_freq": doc_freq,
            "mean_attr": mean_attr,
            "weighted_score": weighted_score,
        })

    df = pd.DataFrame(rows_out)

    if df.empty:
        print("[WARN] Aggregation produced an empty dataframe.")
        print(f"[INFO] Rows processed: {n_rows}")
        print(f"[INFO] Rows missing profession key: {n_rows_missing_group}")
        print(f"[INFO] Tokens seen: {n_tokens_seen}")
        print(f"[INFO] Tokens kept after filtering: {n_tokens_kept}")
        return df

    sort_cols = ["profession", "weighted_score"]
    ascending = [True, False]

    if group_by_gender:
        sort_cols = ["profession", "gender", "weighted_score"]
        ascending = [True, True, False]

    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    print(f"[INFO] Rows processed: {n_rows}")
    print(f"[INFO] Rows missing profession key: {n_rows_missing_group}")
    print(f"[INFO] Tokens seen: {n_tokens_seen}")
    print(f"[INFO] Tokens kept after filtering: {n_tokens_kept}")
    print(f"[INFO] Aggregated rows: {len(df)}")

    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Saved CSV: {path}")


def print_top_tokens_per_profession(
    df: pd.DataFrame,
    top_n: int = 10,
    group_by_gender: bool = False,
) -> None:
    if df.empty:
        print("[WARN] No data to print.")
        return

    if group_by_gender:
        grouped = df.groupby(["profession", "gender"], sort=False)
        for (profession, gender), sub in grouped:
            print("=" * 80)
            print(f"{profession} | gender={gender}")
            print(sub.head(top_n)[["token", "count_topk", "doc_freq", "mean_attr", "weighted_score"]].to_string(index=False))
    else:
        grouped = df.groupby("profession", sort=False)
        for profession, sub in grouped:
            print("=" * 80)
            print(profession)
            print(sub.head(top_n)[["token", "count_topk", "doc_freq", "mean_attr", "weighted_score"]].to_string(index=False))