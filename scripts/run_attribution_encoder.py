from __future__ import annotations

import argparse

from src.attribution.attribution_encoder import run_attribution_on_jsonl


def main():
    parser = argparse.ArgumentParser(
        description="Run Integrated Gradients attribution on encoder prediction JSONL."
    )
    parser.add_argument("--input_jsonl", required=True, help="Filtered 3000-row prediction JSONL with text")
    parser.add_argument("--model_path", required=True, help="Checkpoint folder from Person 4")
    parser.add_argument("--output_jsonl", required=True, help="Where to save attribution results")
    parser.add_argument("--limit", type=int, default=50, help="How many examples to process")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", default=None, help="cpu / cuda / mps")

    args = parser.parse_args()

    run_attribution_on_jsonl(
        input_jsonl=args.input_jsonl,
        model_path=args.model_path,
        output_jsonl=args.output_jsonl,
        limit=args.limit,
        max_length=args.max_length,
        n_steps=args.n_steps,
        top_k=args.top_k,
        device=args.device,
    )


if __name__ == "__main__":
    main()