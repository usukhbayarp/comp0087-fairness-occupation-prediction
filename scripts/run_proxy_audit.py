from __future__ import annotations

import argparse

from src.attribution.proxy_audit import (
    load_jsonl,
    aggregate_proxy_tokens,
    save_csv,
    print_top_tokens_per_profession,
)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate attribution top tokens into proxy-word audit tables."
    )
    parser.add_argument("--input_jsonl", required=True, help="Attribution JSONL file")
    parser.add_argument("--output_csv", required=True, help="Output CSV for profession-level aggregation")
    parser.add_argument(
        "--output_gender_csv",
        default=None,
        help="Optional output CSV for profession+gender aggregation"
    )
    parser.add_argument("--top_n_print", type=int, default=10, help="How many top rows to print per group")
    parser.add_argument("--min_token_len", type=int, default=3)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--min_doc_freq", type=int, default=1)
    parser.add_argument(
        "--keep_stopwords",
        action="store_true",
        help="Keep stopwords instead of filtering them out"
    )

    args = parser.parse_args()

    rows = load_jsonl(args.input_jsonl)

    # Profession-level aggregation
    df_prof = aggregate_proxy_tokens(
        attribution_rows=rows,
        group_by_gender=False,
        min_token_len=args.min_token_len,
        remove_stopwords=not args.keep_stopwords,
        min_count=args.min_count,
        min_doc_freq=args.min_doc_freq,
    )
    save_csv(df_prof, args.output_csv)
    print_top_tokens_per_profession(df_prof, top_n=args.top_n_print, group_by_gender=False)

    # Optional profession+gender aggregation
    if args.output_gender_csv:
        df_prof_gender = aggregate_proxy_tokens(
            attribution_rows=rows,
            group_by_gender=True,
            min_token_len=args.min_token_len,
            remove_stopwords=not args.keep_stopwords,
            min_count=args.min_count,
            min_doc_freq=args.min_doc_freq,
        )
        save_csv(df_prof_gender, args.output_gender_csv)
        print_top_tokens_per_profession(df_prof_gender, top_n=args.top_n_print, group_by_gender=True)


if __name__ == "__main__":
    main()