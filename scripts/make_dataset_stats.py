# make_dataset_stats.py
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Any

import matplotlib.pyplot as plt

from src.data.preprocessing import load_bios, BiosConfig


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_stats(dataset_dict, label2id) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"splits": {}}

    # overall from train+val+test combined (after filtering)
    overall_gender = Counter()
    overall_label = Counter()
    label_gender = defaultdict(Counter)

    for split_name, ds in dataset_dict.items():
        split_gender = Counter(ds["gender"]) if ds.num_rows else Counter()
        split_label = Counter(ds["label"]) if ds.num_rows else Counter()

        stats["splits"][split_name] = {
            "num_samples": ds.num_rows,
            "gender_counts": dict(split_gender),
            "label_counts": dict(split_label),
        }

        overall_gender.update(split_gender)
        overall_label.update(split_label)
        for g, lab in zip(ds["gender"], ds["label"]):
            label_gender[lab][g] += 1

    total = sum(overall_gender.values()) if overall_gender else 0
    gender_dist = {k: (v / total if total else 0.0) for k, v in overall_gender.items()}

    # class imbalance report
    occ_counts_sorted = sorted(overall_label.items(), key=lambda x: x[1], reverse=True)

    # per-occupation gender distribution
    occ_gender_breakdown = {}
    for lab, cnt in overall_label.items():
        gcounts = dict(label_gender[lab])
        denom = sum(gcounts.values()) if gcounts else 0
        gdist = {k: (v / denom if denom else 0.0) for k, v in gcounts.items()}
        occ_gender_breakdown[lab] = {
            "counts": gcounts,
            "dist": gdist,
            "total": denom
        }

    stats.update({
        "num_labels": len(label2id),
        "num_samples_total": sum(overall_label.values()),
        "gender_counts_overall": dict(overall_gender),
        "gender_distribution_overall": gender_dist,
        "occupation_counts_overall": dict(overall_label),
        "occupation_counts_sorted": occ_counts_sorted,
        "occupation_gender_breakdown": occ_gender_breakdown,
    })

    return stats


def save_csvs(stats: Dict[str, Any], out_dir: str) -> None:
    # occupation_counts.csv
    occ_sorted = stats["occupation_counts_sorted"]
    with open(os.path.join(out_dir, "occupation_counts.csv"), "w", encoding="utf-8") as f:
        f.write("occupation,count\n")
        for lab, cnt in occ_sorted:
            f.write(f"{lab},{cnt}\n")

    # occupation_gender_breakdown.csv
    with open(os.path.join(out_dir, "occupation_gender_breakdown.csv"), "w", encoding="utf-8") as f:
        f.write("occupation,total,M,F,UNK,M_frac,F_frac,UNK_frac\n")
        for lab, info in stats["occupation_gender_breakdown"].items():
            total = info["total"]
            c = info["counts"]
            d = info["dist"]
            f.write(
                f"{lab},{total},{c.get('M',0)},{c.get('F',0)},{c.get('UNK',0)},"
                f"{d.get('M',0.0)},{d.get('F',0.0)},{d.get('UNK',0.0)}\n"
            )


def plot_basic(stats: Dict[str, Any], out_dir: str, top_k: int = 20) -> None:
    # 1) occupation_counts.png (top_k)
    occ_sorted = stats["occupation_counts_sorted"][:top_k]
    labels = [x[0] for x in occ_sorted]
    counts = [x[1] for x in occ_sorted]

    plt.figure()
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
    plt.title(f"Top-{top_k} occupation counts")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "occupation_counts.png"))
    plt.close()

    # 2) gender_overall.png
    gcounts = stats["gender_counts_overall"]
    glabels = list(gcounts.keys())
    gvals = [gcounts[k] for k in glabels]

    plt.figure()
    plt.bar(range(len(glabels)), gvals)
    plt.xticks(range(len(glabels)), glabels, rotation=0)
    plt.title("Gender counts (overall)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gender_overall.png"))
    plt.close()


def main():
    out_dir = "outputs"
    _ensure_dir(out_dir)

    # You can edit these defaults:
    cfg = BiosConfig(
        top_n=20,          # set None for full label set (not recommended)
        min_count=1,
        mask_gender=False, # set True to export a masked version
        lowercase_text=False,
    )

    dataset_dict, label2id, id2label, meta = load_bios(cfg=cfg, splits=("train","dev","test"))

    stats = compute_stats(dataset_dict, label2id)
    payload = {"meta": meta, "stats": stats}

    with open(os.path.join(out_dir, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    save_csvs(stats, out_dir)
    plot_basic(stats, out_dir, top_k=min(20, len(label2id)))

    print(f"Saved: {os.path.join(out_dir, 'dataset_stats.json')}")
    print(f"Splits: {list(dataset_dict.keys())}")
    print(f"Num labels: {len(label2id)}")


if __name__ == "__main__":
    main()