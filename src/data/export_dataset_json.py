import json
import os

from data import load_bios, BiosConfig


def export_dataset_json(dataset, output_dir="exports"):

    os.makedirs(output_dir, exist_ok=True)

    for split, data in dataset.items():

        path = os.path.join(output_dir, f"{split}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {split} dataset -> {path}")


def export_dataset_jsonl(dataset, output_dir="exports"):

    os.makedirs(output_dir, exist_ok=True)

    for split, data in dataset.items():

        path = os.path.join(output_dir, f"{split}.jsonl")

        with open(path, "w", encoding="utf-8") as f:

            for row in data:
                f.write(json.dumps(row) + "\n")

        print(f"Saved {split} dataset -> {path}")


def main():

    cfg = BiosConfig(
        top_n=20,
        mask_gender=False
    )

    dataset, label2id = load_bios(cfg)

    print("Dataset loaded")
    print("Train size:", len(dataset["train"]))
    print("Dev size:", len(dataset["dev"]))
    print("Test size:", len(dataset["test"]))

    # export JSON
    export_dataset_json(dataset)


if __name__ == "__main__":
    main()