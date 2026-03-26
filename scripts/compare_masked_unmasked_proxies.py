from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


def _fallback_gender_token_lists() -> tuple[list[str], list[str], list[str]]:
    gender_terms = [
        "he", "she",
        "his", "her",
        "him", "hers",
        "himself", "herself",
    ]

    title_terms = [
        "mr", "mr.", "mister",
        "mrs", "mrs.", "miss", "ms", "ms.",
        "madam", "madame", "ma'am",
        "sir", "lady", "lord", "dame",
    ]

    gendered_noun_terms = [
        "father", "mother", "dad", "mom", "son", "daughter",
        "brother", "sister", "husband", "wife", "boyfriend", "girlfriend",
        "uncle", "aunt", "nephew", "niece", "grandfather", "grandmother",
        "grandson", "granddaughter", "stepfather", "stepmother", "stepson", "stepdaughter",
        "businessman", "businesswoman", "chairman", "chairwoman",
        "salesman", "saleswoman", "spokesman", "spokeswoman",
        "congressman", "congresswoman", "policeman", "policewoman",
        "fireman", "stewardess", "waiter", "waitress", "actor", "actress",
        "host", "hostess", "king", "queen", "prince", "princess",
        "monk", "nun", "bride", "groom", "widow", "widower",
    ]

    return gender_terms, title_terms, gendered_noun_terms


def load_masking_token_sets() -> dict[str, set[str]]:
    """
    Try to import the exact masking categories from src.data.masking.
    Falls back to a copied list if import is unavailable.
    """
    try:
        from src.data.masking import (
            _GENDER_TERMS,
            _TITLE_TERMS,
            _GENDERED_NOUN_TERMS,
        )

        gender_terms = list(_GENDER_TERMS)
        title_terms = list(_TITLE_TERMS)
        gendered_noun_terms = list(_GENDERED_NOUN_TERMS)

    except Exception:
        gender_terms, title_terms, gendered_noun_terms = _fallback_gender_token_lists()

    def normalize_terms(terms: Iterable[str]) -> set[str]:
        out = set()
        for t in terms:
            if not t:
                continue
            t = str(t).strip().lower()
            out.add(t)
            # also include punctuation-stripped form
            out.add(t.replace(".", ""))
            out.add(t.replace("'", ""))
        return out

    return {
        "pronouns": normalize_terms(gender_terms),
        "titles": normalize_terms(title_terms),
        "gendered_nouns": normalize_terms(gendered_noun_terms),
        "all_gender_tokens": normalize_terms(
            list(gender_terms) + list(title_terms) + list(gendered_noun_terms)
        ),
    }


def load_proxy_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"profession", "token", "weighted_score", "count_topk", "doc_freq", "mean_attr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def load_proxy_gender_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"profession", "gender", "token", "weighted_score", "count_topk", "doc_freq", "mean_attr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def top_tokens_by_profession(df: pd.DataFrame, top_n: int) -> dict[str, pd.DataFrame]:
    out = {}
    for profession, sub in df.groupby("profession", sort=False):
        out[profession] = sub.sort_values("weighted_score", ascending=False).head(top_n).copy()
    return out


def compare_top_tokens(masked: pd.DataFrame, unmasked: pd.DataFrame, top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    masked_top = top_tokens_by_profession(masked, top_n)
    unmasked_top = top_tokens_by_profession(unmasked, top_n)

    professions = sorted(set(masked["profession"]).union(set(unmasked["profession"])))

    summary_rows = []
    token_shift_rows = []

    for profession in professions:
        m = masked_top.get(profession, pd.DataFrame(columns=masked.columns))
        u = unmasked_top.get(profession, pd.DataFrame(columns=unmasked.columns))

        m_tokens = set(m["token"].astype(str).tolist())
        u_tokens = set(u["token"].astype(str).tolist())

        shared = sorted(m_tokens & u_tokens)
        masked_only = sorted(m_tokens - u_tokens)
        unmasked_only = sorted(u_tokens - m_tokens)

        summary_rows.append({
            "profession": profession,
            "n_shared_top_tokens": len(shared),
            "n_masked_only_top_tokens": len(masked_only),
            "n_unmasked_only_top_tokens": len(unmasked_only),
            "shared_tokens": ", ".join(shared),
            "masked_only_tokens": ", ".join(masked_only),
            "unmasked_only_tokens": ", ".join(unmasked_only),
        })

        merged = (
            u[["token", "weighted_score", "count_topk", "doc_freq", "mean_attr"]]
            .rename(columns={
                "weighted_score": "weighted_score_unmasked",
                "count_topk": "count_topk_unmasked",
                "doc_freq": "doc_freq_unmasked",
                "mean_attr": "mean_attr_unmasked",
            })
            .merge(
                m[["token", "weighted_score", "count_topk", "doc_freq", "mean_attr"]]
                .rename(columns={
                    "weighted_score": "weighted_score_masked",
                    "count_topk": "count_topk_masked",
                    "doc_freq": "doc_freq_masked",
                    "mean_attr": "mean_attr_masked",
                }),
                on="token",
                how="outer",
            )
            .fillna(0.0)
        )

        merged["profession"] = profession
        merged["weighted_score_diff_masked_minus_unmasked"] = (
            merged["weighted_score_masked"] - merged["weighted_score_unmasked"]
        )

        token_shift_rows.append(
            merged[[
                "profession",
                "token",
                "weighted_score_unmasked",
                "weighted_score_masked",
                "weighted_score_diff_masked_minus_unmasked",
                "count_topk_unmasked",
                "count_topk_masked",
                "doc_freq_unmasked",
                "doc_freq_masked",
                "mean_attr_unmasked",
                "mean_attr_masked",
            ]]
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("profession").reset_index(drop=True)
    token_shift_df = pd.concat(token_shift_rows, ignore_index=True).sort_values(
        ["profession", "weighted_score_diff_masked_minus_unmasked"],
        ascending=[True, False]
    )

    return summary_df, token_shift_df


def compare_explicit_gender_tokens(
    masked: pd.DataFrame,
    unmasked: pd.DataFrame,
    token_set: set[str],
    category_name: str,
) -> pd.DataFrame:
    """
    Compare explicit gender-token presence between masked and unmasked proxy tables.
    """
    def subset(df: pd.DataFrame, name: str) -> pd.DataFrame:
        sub = df[df["token"].astype(str).str.lower().isin(token_set)].copy()
        grouped = sub.groupby("token", as_index=False).agg({
            "count_topk": "sum",
            "doc_freq": "sum",
            "weighted_score": "sum",
            "mean_attr": "mean",
        })
        grouped = grouped.rename(columns={
            "count_topk": f"count_topk_{name}",
            "doc_freq": f"doc_freq_{name}",
            "weighted_score": f"weighted_score_{name}",
            "mean_attr": f"mean_attr_{name}",
        })
        return grouped

    u = subset(unmasked, "unmasked")
    m = subset(masked, "masked")

    out = u.merge(m, on="token", how="outer").fillna(0.0)
    out["category"] = category_name
    out["weighted_score_diff_masked_minus_unmasked"] = (
        out["weighted_score_masked"] - out["weighted_score_unmasked"]
    )
    return out.sort_values("weighted_score_diff_masked_minus_unmasked")


def build_all_explicit_gender_reports(
    masked: pd.DataFrame,
    unmasked: pd.DataFrame,
) -> pd.DataFrame:
    token_sets = load_masking_token_sets()

    dfs = []
    for category in ["pronouns", "titles", "gendered_nouns", "all_gender_tokens"]:
        dfs.append(
            compare_explicit_gender_tokens(
                masked=masked,
                unmasked=unmasked,
                token_set=token_sets[category],
                category_name=category,
            )
        )

    return pd.concat(dfs, ignore_index=True)


def compare_gender_conditioned_proxies(
    masked_gender: pd.DataFrame,
    unmasked_gender: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Compare masked vs unmasked within each (profession, gender) subgroup.
    Produces a summary table, not a token-level giant matrix.
    """
    summary_rows = []

    keys = sorted(
        set(zip(masked_gender["profession"], masked_gender["gender"])) |
        set(zip(unmasked_gender["profession"], unmasked_gender["gender"]))
    )

    for profession, gender in keys:
        m = masked_gender[
            (masked_gender["profession"] == profession) &
            (masked_gender["gender"] == gender)
        ].sort_values("weighted_score", ascending=False).head(top_n)

        u = unmasked_gender[
            (unmasked_gender["profession"] == profession) &
            (unmasked_gender["gender"] == gender)
        ].sort_values("weighted_score", ascending=False).head(top_n)

        m_tokens = set(m["token"].astype(str).tolist())
        u_tokens = set(u["token"].astype(str).tolist())

        summary_rows.append({
            "profession": profession,
            "gender": gender,
            "n_shared_top_tokens": len(m_tokens & u_tokens),
            "n_masked_only_top_tokens": len(m_tokens - u_tokens),
            "n_unmasked_only_top_tokens": len(u_tokens - m_tokens),
            "shared_tokens": ", ".join(sorted(m_tokens & u_tokens)),
            "masked_only_tokens": ", ".join(sorted(m_tokens - u_tokens)),
            "unmasked_only_tokens": ", ".join(sorted(u_tokens - m_tokens)),
        })

    return pd.DataFrame(summary_rows).sort_values(["profession", "gender"]).reset_index(drop=True)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masked_csv", required=True)
    parser.add_argument("--unmasked_csv", required=True)

    parser.add_argument("--out_summary_csv", required=True)
    parser.add_argument("--out_token_shift_csv", required=True)
    parser.add_argument("--out_gender_token_csv", required=True)

    parser.add_argument("--top_n", type=int, default=10)

    # Optional gender-conditioned comparison
    parser.add_argument("--masked_gender_csv", default=None)
    parser.add_argument("--unmasked_gender_csv", default=None)
    parser.add_argument("--out_gender_conditioned_summary_csv", default=None)

    args = parser.parse_args()

    masked = load_proxy_csv(args.masked_csv)
    unmasked = load_proxy_csv(args.unmasked_csv)

    summary_df, token_shift_df = compare_top_tokens(masked, unmasked, top_n=args.top_n)
    explicit_gender_df = build_all_explicit_gender_reports(masked, unmasked)

    output_items = [
        (args.out_summary_csv, summary_df),
        (args.out_token_shift_csv, token_shift_df),
        (args.out_gender_token_csv, explicit_gender_df),
    ]

    # Optional profession+gender comparison
    if args.masked_gender_csv and args.unmasked_gender_csv and args.out_gender_conditioned_summary_csv:
        masked_gender = load_proxy_gender_csv(args.masked_gender_csv)
        unmasked_gender = load_proxy_gender_csv(args.unmasked_gender_csv)

        gender_conditioned_df = compare_gender_conditioned_proxies(
            masked_gender=masked_gender,
            unmasked_gender=unmasked_gender,
            top_n=args.top_n,
        )
        output_items.append((args.out_gender_conditioned_summary_csv, gender_conditioned_df))

    for path_str, df in output_items:
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    print(f"Saved summary: {args.out_summary_csv}")
    print(f"Saved token shifts: {args.out_token_shift_csv}")
    print(f"Saved explicit gender-token comparison: {args.out_gender_token_csv}")

    if args.out_gender_conditioned_summary_csv:
        print(f"Saved profession+gender masked-vs-unmasked summary: {args.out_gender_conditioned_summary_csv}")

    print("\nSample profession-level summary:")
    print(summary_df.head(10).to_string(index=False))

    print("\nSample explicit gender-token comparison:")
    print(explicit_gender_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()