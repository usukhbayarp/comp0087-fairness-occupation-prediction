# data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List
from collections import Counter
from pathlib import Path
import json

from datasets import load_dataset, Dataset, DatasetDict

from masking import (
    mask_gender_terms,
    mask_titles,
    mask_gendered_nouns,
    mask_label_leakage,
    contains_label_leakage,
)


@dataclass(frozen=True)
class BiosConfig:
    dataset_name: str = "LabHC/bias_in_bios"

    # Label policy
    top_n: Optional[int] = 20
    min_count: int = 1

    # Text processing
    lowercase_text: bool = False
    mask_gender: bool = False
    mask_titles: bool = False
    mask_gendered_nouns: bool = False
    mask_label_leakage: bool = False
    mask_token: str = "[MASK]"
    label_mask_token: str = "[LABEL_MASK]"

    # Gender normalization output
    unk_gender: str = "UNK"

    # Optional profession-id -> profession-name mapping JSON path
    #profession_mapping_path: Optional[str] = str(Path(__file__).with_name("profession_mapping.json"))
    profession_mapping_path: Optional[str] = str(
        Path(__file__).resolve().parents[2] / "data" / "profession_mapping.json")

    # If you already KNOW the mapping, you can hard-set it here, e.g. {0:"M",1:"F"}
    # Leave None to auto-infer from observed ids in train split.
    gender_id_mapping_override: Optional[Dict[int, str]] = None


def _infer_columns(ds: Dataset) -> Dict[str, str]:
    cols = set(ds.column_names)

    text_candidates = ["hard_text", "bio", "biography", "text"]
    label_candidates = ["profession", "occupation", "label"]
    gender_candidates = ["gender", "sex"]

    text_col = next((c for c in text_candidates if c in cols), None)
    label_col = next((c for c in label_candidates if c in cols), None)
    gender_col = next((c for c in gender_candidates if c in cols), None)

    if text_col is None:
        raise ValueError(f"Cannot find text column in {ds.column_names}")
    if label_col is None:
        raise ValueError(f"Cannot find label column in {ds.column_names}")
    if gender_col is None:
        raise ValueError(f"Cannot find gender column in {ds.column_names}")

    # id optional; generate if absent
    id_col = next((c for c in ["id", "uid", "guid", "index"] if c in cols), "")

    return {"text": text_col, "label": label_col, "gender": gender_col, "id": id_col}


def _get_classlabel_names(ds: Dataset, col: str) -> Optional[List[str]]:
    feat = ds.features.get(col, None)
    names = getattr(feat, "names", None)
    return list(names) if names is not None else None


def _to_label_string(raw_val: Any, names: Optional[List[str]]) -> str:
    if raw_val is None:
        return ""
    if isinstance(raw_val, str):
        return raw_val
    if names is not None:
        try:
            return names[int(raw_val)]
        except Exception:
            return str(raw_val)
    return str(raw_val)




def _load_profession_mapping(mapping_path: Optional[str]) -> Dict[str, str]:
    if not mapping_path:
        return {}
    try:
        path = Path(mapping_path)
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}


def _normalize_profession_name(name: str) -> str:
    s = str(name).strip()
    return s.replace("_", " ")


def _map_label_value(raw_val: Any, names: Optional[List[str]], profession_mapping: Optional[Dict[str, str]] = None) -> str:
    label = _to_label_string(raw_val, names)
    if profession_mapping:
        mapped = profession_mapping.get(label)
        if mapped is None:
            # handle values like 21 / "21" robustly
            try:
                mapped = profession_mapping.get(str(int(label)))
            except Exception:
                mapped = None
        if mapped is not None:
            return _normalize_profession_name(mapped)
    return _normalize_profession_name(label)

def _infer_gender_id_mapping_from_train(
    train_gender_values: List[Any],
) -> Tuple[Optional[Dict[int, str]], Dict[str, Any]]:
    """
    Try to infer mapping from numeric gender ids to {"M","F","UNK"}.

    Common cases:
      - {0,1}  -> 0:M, 1:F   (assumption; can be flipped if dataset defines otherwise)
      - {1,2}  -> 1:M, 2:F
      - {0,1,2} -> 0:M, 1:F, 2:UNK (or similar)

    Returns:
      (mapping or None if cannot infer, debug_meta)
    """
    uniq = set()
    for v in train_gender_values:
        # keep only int-like
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            uniq.add(v)
        else:
            # sometimes stored as numpy int / string digit
            try:
                iv = int(v)
                uniq.add(iv)
            except Exception:
                pass

    debug = {"observed_gender_unique_ints": sorted(list(uniq))}

    if not uniq:
        return None, debug

    # Heuristic mappings
    if uniq == {0, 1}:
        return {0: "M", 1: "F"}, debug
    if uniq == {1, 2}:
        return {1: "M", 2: "F"}, debug
    if uniq == {0, 1, 2}:
        # third id treated as UNK
        return {0: "M", 1: "F", 2: "UNK"}, debug
    if uniq == {1, 2, 3}:
        return {1: "M", 2: "F", 3: "UNK"}, debug

    # If there are exactly two ids but not standard ones, map smaller->M larger->F
    if len(uniq) == 2:
        a, b = sorted(list(uniq))
        return {a: "M", b: "F"}, debug

    return None, debug


def _normalize_gender(
    raw_gender: Any,
    gender_names: Optional[List[str]],
    gender_id_mapping: Optional[Dict[int, str]],
    unk: str = "UNK",
) -> str:
    """
    Normalize gender to {"M","F","UNK"}.

    Priority:
      1) If ClassLabel names exist -> map id->name -> string rules
      2) Else if numeric mapping inferred/provided -> map int id to M/F/UNK
      3) Else string rules on raw_gender
    """
    if raw_gender is None:
        return unk

    # 1) ClassLabel mapping if present
    if not isinstance(raw_gender, str) and gender_names is not None:
        try:
            raw_gender = gender_names[int(raw_gender)]
        except Exception:
            raw_gender = str(raw_gender)

    # 2) numeric mapping
    if gender_id_mapping is not None and not isinstance(raw_gender, str):
        try:
            gid = int(raw_gender)
            mapped = gender_id_mapping.get(gid, None)
            if mapped in {"M", "F", "UNK"}:
                return mapped
        except Exception:
            pass

    # Also handle string digits "0"/"1"/"2"
    if gender_id_mapping is not None and isinstance(raw_gender, str):
        s0 = raw_gender.strip()
        if s0.isdigit():
            gid = int(s0)
            mapped = gender_id_mapping.get(gid, None)
            if mapped in {"M", "F", "UNK"}:
                return mapped

    # 3) string rules
    s = str(raw_gender).strip().lower()
    if s in {"m", "male", "man", "masculine"}:
        return "M"
    if s in {"f", "female", "woman", "feminine"}:
        return "F"
    return unk


def build_label_vocab_from_strings(
    train_label_strings: List[str],
    top_n: Optional[int],
    min_count: int
) -> Tuple[Dict[str, int], Dict[int, str], List[Tuple[str, int]]]:
    counts = Counter(train_label_strings)
    items = [(lab, cnt) for lab, cnt in counts.items() if lab != "" and cnt >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)

    if top_n is not None:
        items = items[:top_n]

    kept = [lab for lab, _ in items]
    label2id = {lab: i for i, lab in enumerate(kept)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label, items


def load_bios(
    cfg: Optional[BiosConfig] = None,
    splits: Tuple[str, ...] = ("train", "dev", "test"),
) -> Tuple[DatasetDict, Dict[str, int], Dict[int, str], Dict[str, Any]]:
    cfg = cfg or BiosConfig()
    raw: DatasetDict = load_dataset(cfg.dataset_name)

    if "train" not in raw:
        raise ValueError(f"Dataset has no train split. Available: {list(raw.keys())}")

    field_map = _infer_columns(raw["train"])
    text_col = field_map["text"]
    label_col = field_map["label"]
    gender_col = field_map["gender"]
    id_col = field_map["id"]

    # ClassLabel names if present (your stats show both are False) :contentReference[oaicite:1]{index=1}
    label_names = _get_classlabel_names(raw["train"], label_col)
    gender_names = _get_classlabel_names(raw["train"], gender_col)

    # Infer numeric gender mapping if needed
    gender_mapping_meta: Dict[str, Any] = {}
    if cfg.gender_id_mapping_override is not None:
        gender_id_mapping = dict(cfg.gender_id_mapping_override)
        gender_mapping_meta["gender_id_mapping_source"] = "override"
        gender_mapping_meta["gender_id_mapping"] = gender_id_mapping
    else:
        inferred_map, debug = _infer_gender_id_mapping_from_train(raw["train"][gender_col])
        gender_id_mapping = inferred_map
        gender_mapping_meta["gender_id_mapping_source"] = "inferred"
        gender_mapping_meta.update(debug)
        gender_mapping_meta["gender_id_mapping"] = gender_id_mapping

    profession_mapping = _load_profession_mapping(cfg.profession_mapping_path)

    # Build label vocab from TRAIN using profession names (mapped from ids when possible)
    train_label_strings = [_map_label_value(x, label_names, profession_mapping) for x in raw["train"][label_col]]
    label2id, id2label, label_counts_sorted = build_label_vocab_from_strings(
        train_label_strings=train_label_strings,
        top_n=cfg.top_n,
        min_count=cfg.min_count
    )
    kept_labels = set(label2id.keys())

    def _convert(ex: Dict[str, Any]) -> Dict[str, Any]:
        # label (profession id -> profession name when mapping is available)
        label = _map_label_value(ex.get(label_col), label_names, profession_mapping)

        # text
        text = "" if ex.get(text_col) is None else str(ex.get(text_col))
        label_leakage_before_mask = contains_label_leakage(text, label)

        if cfg.lowercase_text:
            text = text.lower()
        if cfg.mask_gender:
            text = mask_gender_terms(text, mask_token=cfg.mask_token)
        if cfg.mask_titles:
            text = mask_titles(text, mask_token=cfg.mask_token)
        if cfg.mask_gendered_nouns:
            text = mask_gendered_nouns(text, mask_token=cfg.mask_token)
        if cfg.mask_label_leakage:
            text = mask_label_leakage(text, label, mask_token=cfg.label_mask_token)

        # gender (robust)
        gender = _normalize_gender(
            ex.get(gender_col),
            gender_names=gender_names,
            gender_id_mapping=gender_id_mapping,
            unk=cfg.unk_gender
        )

        # id optional
        rid = ex.get(id_col) if id_col else None

        return {
            "id": rid,
            "text": text,
            "label": label,
            "gender": gender,
            "label_id": label2id.get(label, -1),
            "label_leakage": label_leakage_before_mask,
        }

    processed = DatasetDict()
    for sp in splits:
        if sp not in raw:
            continue

        ds = raw[sp].map(_convert, remove_columns=raw[sp].column_names)

        # filter to top-N labels (based on TRAIN)
        if cfg.top_n is not None:
            ds = ds.filter(lambda x: x["label"] in kept_labels)

        # fill missing ids
        ds = ds.map(lambda x, idx: {"id": idx if x["id"] is None else x["id"]}, with_indices=True)

        processed[sp] = ds

    meta = {
        "dataset_name": cfg.dataset_name,
        "splits_returned": list(processed.keys()),
        "field_map_inferred": field_map,
        "raw_columns_train": raw["train"].column_names,
        "label_col_classlabel": bool(label_names is not None),
        "gender_col_classlabel": bool(gender_names is not None),
        "top_n": cfg.top_n,
        "min_count": cfg.min_count,
        "num_labels": len(label2id),
        "labels": [id2label[i] for i in range(len(id2label))],
        "profession_mapping_path": cfg.profession_mapping_path,
        "profession_mapping_loaded": bool(profession_mapping),
        "label_counts_train_sorted": label_counts_sorted,
        "mask_gender": cfg.mask_gender,
        "mask_titles": cfg.mask_titles,
        "mask_gendered_nouns": cfg.mask_gendered_nouns,
        "mask_label_leakage": cfg.mask_label_leakage,
        "mask_token": cfg.mask_token if (cfg.mask_gender or cfg.mask_titles or cfg.mask_gendered_nouns) else None,
        "label_mask_token": cfg.label_mask_token if cfg.mask_label_leakage else None,
        "lowercase_text": cfg.lowercase_text,
        "gender_normalization": {"M": "male", "F": "female", "UNK": "unknown/other"},
        **gender_mapping_meta,
    }

    return processed, label2id, id2label, meta
