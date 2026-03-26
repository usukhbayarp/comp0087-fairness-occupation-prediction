from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(p)


def parse_token_list(x) -> list[str]:
    if pd.isna(x) or str(x).strip() == "":
        return []
    return [t.strip() for t in str(x).split(",") if t.strip()]


def summarize_profession_overlap(summary_df: pd.DataFrame, top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = summary_df.copy()
    df["shared_ratio"] = df["n_shared_top_tokens"] / (
        df["n_shared_top_tokens"] + df["n_masked_only_top_tokens"] + df["n_unmasked_only_top_tokens"]
    ).replace(0, 1)

    most_stable = df.sort_values(
        ["shared_ratio", "n_shared_top_tokens"], ascending=[False, False]
    ).head(top_n)

    least_stable = df.sort_values(
        ["shared_ratio", "n_shared_top_tokens"], ascending=[True, True]
    ).head(top_n)

    return most_stable, least_stable


def summarize_token_shift(
    token_shift_df: pd.DataFrame,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = token_shift_df.copy()

    # strongest increases after masking
    strongest_increase = df.sort_values(
        "weighted_score_diff_masked_minus_unmasked", ascending=False
    ).head(top_n)

    # strongest decreases after masking
    strongest_decrease = df.sort_values(
        "weighted_score_diff_masked_minus_unmasked", ascending=True
    ).head(top_n)

    # profession-level absolute shift magnitude
    prof_shift = (
        df.assign(abs_shift=df["weighted_score_diff_masked_minus_unmasked"].abs())
        .groupby("profession", as_index=False)["abs_shift"]
        .sum()
        .sort_values("abs_shift", ascending=False)
    )

    return strongest_increase, strongest_decrease, prof_shift


def summarize_gender_conditioned(
    gender_summary_df: pd.DataFrame,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = gender_summary_df.copy()
    df["shared_ratio"] = df["n_shared_top_tokens"] / (
        df["n_shared_top_tokens"] + df["n_masked_only_top_tokens"] + df["n_unmasked_only_top_tokens"]
    ).replace(0, 1)

    most_stable = df.sort_values(
        ["shared_ratio", "n_shared_top_tokens"], ascending=[False, False]
    ).head(top_n)

    least_stable = df.sort_values(
        ["shared_ratio", "n_shared_top_tokens"], ascending=[True, True]
    ).head(top_n)

    return most_stable, least_stable


def summarize_explicit_gender_tokens(gender_tokens_df: pd.DataFrame) -> pd.DataFrame:
    if gender_tokens_df.empty:
        return gender_tokens_df

    df = gender_tokens_df.copy()
    keep_cols = [
        c for c in [
            "category", "token",
            "count_topk_unmasked", "count_topk_masked",
            "doc_freq_unmasked", "doc_freq_masked",
            "weighted_score_unmasked", "weighted_score_masked",
            "weighted_score_diff_masked_minus_unmasked",
        ] if c in df.columns
    ]
    return df[keep_cols].sort_values(
        "weighted_score_diff_masked_minus_unmasked"
    )


def write_text_report(
    out_path: str,
    model_name: str,
    most_stable_prof: pd.DataFrame,
    least_stable_prof: pd.DataFrame,
    strongest_increase: pd.DataFrame,
    strongest_decrease: pd.DataFrame,
    prof_shift: pd.DataFrame,
    most_stable_gender: pd.DataFrame,
    least_stable_gender: pd.DataFrame,
    gender_tokens_df: pd.DataFrame | None = None,
) -> None:
    lines: list[str] = []
    lines.append(f"Model: {model_name}")
    lines.append("")
    lines.append("=== Most stable professions (masked vs unmasked) ===")
    for _, r in most_stable_prof.iterrows():
        lines.append(
            f"{r['profession']}: shared={r['n_shared_top_tokens']}, "
            f"masked_only={r['n_masked_only_top_tokens']}, "
            f"unmasked_only={r['n_unmasked_only_top_tokens']}, "
            f"shared_ratio={r['shared_ratio']:.2f}"
        )

    lines.append("")
    lines.append("=== Least stable professions (masked vs unmasked) ===")
    for _, r in least_stable_prof.iterrows():
        lines.append(
            f"{r['profession']}: shared={r['n_shared_top_tokens']}, "
            f"masked_only={r['n_masked_only_top_tokens']}, "
            f"unmasked_only={r['n_unmasked_only_top_tokens']}, "
            f"shared_ratio={r['shared_ratio']:.2f}"
        )

    lines.append("")
    lines.append("=== Strongest token increases after masking ===")
    for _, r in strongest_increase.iterrows():
        lines.append(
            f"{r['profession']} | {r['token']}: "
            f"unmasked={r['weighted_score_unmasked']:.3f}, "
            f"masked={r['weighted_score_masked']:.3f}, "
            f"diff={r['weighted_score_diff_masked_minus_unmasked']:.3f}"
        )

    lines.append("")
    lines.append("=== Strongest token decreases after masking ===")
    for _, r in strongest_decrease.iterrows():
        lines.append(
            f"{r['profession']} | {r['token']}: "
            f"unmasked={r['weighted_score_unmasked']:.3f}, "
            f"masked={r['weighted_score_masked']:.3f}, "
            f"diff={r['weighted_score_diff_masked_minus_unmasked']:.3f}"
        )

    lines.append("")
    lines.append("=== Professions with largest overall token shift ===")
    for _, r in prof_shift.head(10).iterrows():
        lines.append(f"{r['profession']}: total_abs_shift={r['abs_shift']:.3f}")

    lines.append("")
    lines.append("=== Most stable profession+gender groups ===")
    for _, r in most_stable_gender.iterrows():
        lines.append(
            f"{r['profession']} | {r['gender']}: "
            f"shared={r['n_shared_top_tokens']}, "
            f"masked_only={r['n_masked_only_top_tokens']}, "
            f"unmasked_only={r['n_unmasked_only_top_tokens']}, "
            f"shared_ratio={r['shared_ratio']:.2f}"
        )

    lines.append("")
    lines.append("=== Least stable profession+gender groups ===")
    for _, r in least_stable_gender.iterrows():
        lines.append(
            f"{r['profession']} | {r['gender']}: "
            f"shared={r['n_shared_top_tokens']}, "
            f"masked_only={r['n_masked_only_top_tokens']}, "
            f"unmasked_only={r['n_unmasked_only_top_tokens']}, "
            f"shared_ratio={r['shared_ratio']:.2f}"
        )

    if gender_tokens_df is not None:
        lines.append("")
        lines.append("=== Explicit gender-token comparison ===")
        if gender_tokens_df.empty:
            lines.append("No explicit gender tokens survived into the final aggregated proxy tables.")
        else:
            for _, r in gender_tokens_df.head(20).iterrows():
                lines.append(
                    f"{r.get('category', 'NA')} | {r['token']}: "
                    f"unmasked={r.get('weighted_score_unmasked', 0):.3f}, "
                    f"masked={r.get('weighted_score_masked', 0):.3f}, "
                    f"diff={r.get('weighted_score_diff_masked_minus_unmasked', 0):.3f}"
                )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--summary_csv", required=True)
    parser.add_argument("--token_shift_csv", required=True)
    parser.add_argument("--profession_gender_summary_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--gender_tokens_csv", default=None)
    parser.add_argument("--top_n", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df = load_csv(args.summary_csv)
    token_shift_df = load_csv(args.token_shift_csv)
    profession_gender_df = load_csv(args.profession_gender_summary_csv)
    gender_tokens_df = load_csv(args.gender_tokens_csv) if args.gender_tokens_csv else None

    most_stable_prof, least_stable_prof = summarize_profession_overlap(summary_df, args.top_n)
    strongest_increase, strongest_decrease, prof_shift = summarize_token_shift(token_shift_df, args.top_n)
    most_stable_gender, least_stable_gender = summarize_gender_conditioned(profession_gender_df, args.top_n)
    gender_tokens_summary = summarize_explicit_gender_tokens(gender_tokens_df) if gender_tokens_df is not None else None

    most_stable_prof.to_csv(out_dir / "most_stable_professions.csv", index=False)
    least_stable_prof.to_csv(out_dir / "least_stable_professions.csv", index=False)
    strongest_increase.to_csv(out_dir / "strongest_token_increases.csv", index=False)
    strongest_decrease.to_csv(out_dir / "strongest_token_decreases.csv", index=False)
    prof_shift.to_csv(out_dir / "profession_total_shift.csv", index=False)
    most_stable_gender.to_csv(out_dir / "most_stable_profession_gender.csv", index=False)
    least_stable_gender.to_csv(out_dir / "least_stable_profession_gender.csv", index=False)

    if gender_tokens_summary is not None:
        gender_tokens_summary.to_csv(out_dir / "explicit_gender_token_summary.csv", index=False)

    write_text_report(
        out_path=str(out_dir / "analysis_report.txt"),
        model_name=args.model_name,
        most_stable_prof=most_stable_prof,
        least_stable_prof=least_stable_prof,
        strongest_increase=strongest_increase,
        strongest_decrease=strongest_decrease,
        prof_shift=prof_shift,
        most_stable_gender=most_stable_gender,
        least_stable_gender=least_stable_gender,
        gender_tokens_df=gender_tokens_summary,
    )

    print(f"Saved analysis outputs to: {out_dir}")


if __name__ == "__main__":
    main()