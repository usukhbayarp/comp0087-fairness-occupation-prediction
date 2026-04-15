from __future__ import annotations

import argparse

from src.attribution.counterfactual_swap import run_counterfactual_swap


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactual pronoun swap experiment. "
            "Swaps gendered pronouns in each biography (M→F or F→M), "
            "re-runs the model, and measures occupation prediction flip rates."
        )
    )
    parser.add_argument(
        "--input_jsonl", required=True,
        help=(
            "Prediction JSONL file. "
            "Required fields: id, text, label_pred, gender. "
            "Use results/predictions/<model>_<condition>.jsonl."
        ),
    )
    parser.add_argument(
        "--model_path", required=True,
        help="Path to fine-tuned encoder checkpoint directory.",
    )
    parser.add_argument(
        "--output_csv", required=True,
        help=(
            "Output path for per-example results CSV. "
            "Two additional summary CSVs are written automatically: "
            "<stem>_by_profession.csv and <stem>_by_profession_gender.csv."
        ),
    )
    parser.add_argument(
        "--max_length", type=int, default=256,
        help="Tokenisation truncation length (default: 256).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N examples (default: all).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: cpu / cuda / mps (auto-detected if not specified).",
    )

    args = parser.parse_args()

    run_counterfactual_swap(
        input_jsonl=args.input_jsonl,
        model_path=args.model_path,
        output_csv=args.output_csv,
        max_length=args.max_length,
        limit=args.limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()