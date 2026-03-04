import os
import json
import argparse
from src.data.preprocessing import load_bios, BiosConfig

def export_jsonl(output_dir="processed", top_n=20, mask_gender=False):
    os.makedirs(output_dir, exist_ok=True)
    
    # Optional parameters can be found in data.py, and use example can be found in make_dataset_stats.py
    cfg = BiosConfig(
        top_n=top_n,
        mask_gender=mask_gender
    )
    
    print("Loading and processing dataset via data.py...")
    try:
        dataset_dict, label2id, id2label, meta = load_bios(cfg=cfg)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    # Load profession mapping
    mapping_path = os.path.join(os.path.dirname(__file__), "profession_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as fm:
        profession_mapping = json.load(fm)

    candidate_labels = [profession_mapping.get(str(id2label[i]), str(id2label[i])) for i in range(len(id2label))] # Maps the ids to the profession names

    # Exports the dataset to JSONL in /processed
    for split_name, ds in dataset_dict.items():
        out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        print(f"Exporting {split_name} split to {out_path} ({len(ds)} records)...")
        
        with open(out_path, "w", encoding="utf-8") as f:
            for item in ds:
                record = {
                    "id": item["id"],
                    "text": item["text"],
                    "label": profession_mapping.get(str(item["label"])),
                    "gender": item["gender"]
                }
                f.write(json.dumps(record) + "\n")
                
    # Save the labels out for the prompt engineering part
    with open(os.path.join(output_dir, "candidate_labels.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(candidate_labels))
        
    print(f"\nExport complete! JSONL files are in '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export HuggingFace DatasetDict to JSONL for Part 2 scripts.")
    parser.add_argument("--output_dir", type=str, default="processed", help="Directory to save JSONL files")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top occupations to retain")
    parser.add_argument("--mask_gender", action="store_true", help="Whether to apply gender masking to text")
    
    args = parser.parse_args()
    export_jsonl(args.output_dir, args.top_n, args.mask_gender)
