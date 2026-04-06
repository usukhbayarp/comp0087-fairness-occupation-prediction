import sys
import os
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = get_project_root()
DATA_MODULE_DIR = PROJECT_ROOT / "src" / "data"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DATA_MODULE_DIR))

import json
import argparse
from src.data.data import load_bios, BiosConfig


def resolve_output_dir(output_dir: str) -> Path:
    out_path = Path(output_dir)
    if not out_path.is_absolute():
        if output_dir == "processed":
            out_path = PROJECT_ROOT / "data" / "processed"
        else:
            out_path = PROJECT_ROOT / out_path
    return out_path


def export_jsonl(output_dir="processed", top_n=20, mask_gender=False):
    output_path = resolve_output_dir(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cfg = BiosConfig(
        top_n=top_n,
        mask_gender=mask_gender,
        profession_mapping_path=str(PROJECT_ROOT / "data" / "profession_mapping.json"),
    )

    print("Loading and processing dataset via data.py...")
    try:
        dataset_dict, label2id, id2label, meta = load_bios(cfg=cfg)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    candidate_labels = [str(id2label[i]) for i in range(len(id2label))]

    # Exports the dataset to JSONL in /processed
    for split_name, ds in dataset_dict.items():
        out_path = output_path / f"{split_name}.jsonl"
        print(f"Exporting {split_name} split to {out_path} ({len(ds)} records)...")

        with open(out_path, "w", encoding="utf-8") as f:
            for item in ds:
                record = {
                    "id": item["id"],
                    "text": item["text"],
                    "label": item["label"],
                    "gender": item["gender"]
                }
                f.write(json.dumps(record) + "\n")

    # Save the labels out for the prompt engineering part
    with open(output_path / "candidate_labels.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(candidate_labels))

    print(f"\nExport complete! JSONL files are in '{output_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export HuggingFace DatasetDict to JSONL for Part 2 scripts.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed",
        help="Directory to save JSONL files. Relative path `processed` maps to <repo>/data/processed.",
    )
    parser.add_argument("--top_n", type=int, default=20, help="Number of top occupations to retain")
    parser.add_argument("--mask_gender", action="store_true", help="Whether to apply gender masking to text")

    args = parser.parse_args()
    export_jsonl(args.output_dir, args.top_n, args.mask_gender)
