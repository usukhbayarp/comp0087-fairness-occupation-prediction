# make_dataset_stats.py
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Any

import matplotlib.pyplot as plt

from data import load_bios, BiosConfig


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_stats(dataset_dict, label2id) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"splits": {}}

    # overall from train+val+test combined (after filtering)
    overall_gender = Counter()
    overall_label = Counter()
    overall_leakage = Counter()
    label_gender = defaultdict(Counter)
    label_leakage = defaultdict(Counter)

    for split_name, ds in dataset_dict.items():
        split_gender = Counter(ds["gender"]) if ds.num_rows else Counter()
        split_label = Counter(ds["label"]) if ds.num_rows else Counter()

        split_leakage = Counter(ds["label_leakage"]) if ds.num_rows and "label_leakage" in ds.column_names else Counter()

        stats["splits"][split_name] = {
            "num_samples": ds.num_rows,
            "gender_counts": dict(split_gender),
            "label_counts": dict(split_label),
            "label_leakage_counts": dict(split_leakage),
            "label_leakage_rate": (split_leakage.get(True, 0) / ds.num_rows) if ds.num_rows else 0.0,
        }

        overall_gender.update(split_gender)
        overall_label.update(split_label)
        overall_leakage.update(split_leakage)
        for g, lab, leak in zip(ds["gender"], ds["label"], ds["label_leakage"]):
            label_gender[lab][g] += 1
            label_leakage[lab][leak] += 1

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

    overall_num_samples = sum(overall_label.values())
    overall_label_leakage_rate = (overall_leakage.get(True, 0) / overall_num_samples) if overall_num_samples else 0.0

    occupation_leakage_breakdown = {}
    for lab, cnt in overall_label.items():
        leak_counts = dict(label_leakage[lab])
        denom = sum(leak_counts.values()) if leak_counts else 0
        occupation_leakage_breakdown[lab] = {
            "counts": leak_counts,
            "rate": (leak_counts.get(True, 0) / denom) if denom else 0.0,
            "total": denom,
        }

    stats.update({
        "num_labels": len(label2id),
        "num_samples_total": overall_num_samples,
        "gender_counts_overall": dict(overall_gender),
        "gender_distribution_overall": gender_dist,
        "occupation_counts_overall": dict(overall_label),
        "occupation_counts_sorted": occ_counts_sorted,
        "occupation_gender_breakdown": occ_gender_breakdown,
        "label_leakage_counts_overall": dict(overall_leakage),
        "label_leakage_rate_overall": overall_label_leakage_rate,
        "occupation_label_leakage_breakdown": occupation_leakage_breakdown,
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

    # occupation_label_leakage_breakdown.csv
    with open(os.path.join(out_dir, "occupation_label_leakage_breakdown.csv"), "w", encoding="utf-8") as f:
        f.write("occupation,total,has_leakage,no_leakage,leakage_rate\n")
        for lab, info in stats["occupation_label_leakage_breakdown"].items():
            total = info["total"]
            c = info["counts"]
            f.write(
                f"{lab},{total},{c.get(True,0)},{c.get(False,0)},{info.get('rate',0.0)}\n"
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

    # 3) label_leakage_by_occupation.png
    leak_items = sorted(
        stats["occupation_label_leakage_breakdown"].items(),
        key=lambda x: x[1]["rate"],
        reverse=True,
    )[:top_k]
    if leak_items:
        leak_labels = [x[0] for x in leak_items]
        leak_rates = [x[1]["rate"] for x in leak_items]
        plt.figure()
        plt.bar(range(len(leak_labels)), leak_rates)
        plt.xticks(range(len(leak_labels)), leak_labels, rotation=60, ha="right")
        plt.title(f"Top-{len(leak_labels)} occupation label leakage rates")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "label_leakage_by_occupation.png"))
        plt.close()


def main():
    out_dir = "outputs"
    _ensure_dir(out_dir)

    # You can edit these defaults:
    cfg = BiosConfig(
        top_n=20,                 # set None for full label set (not recommended)
        min_count=1,
        mask_gender=True,        # pronouns/possessives
        mask_titles=True,        # e.g. Mr., Sir
        mask_gendered_nouns=True,# e.g. father, actress
        mask_label_leakage=True, # profession words in the bio text
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
