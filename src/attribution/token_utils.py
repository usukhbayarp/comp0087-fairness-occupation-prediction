from __future__ import annotations

from typing import List, Dict, Any


SPECIAL_TOKENS = {
    "[CLS]", "[SEP]", "[PAD]", "[MASK]",
    "<s>", "</s>", "<pad>", "<mask>"
}


def merge_subwords(tokens: List[str], scores: List[float]) -> List[Dict[str, Any]]:
    """
    Merge BERT-style ## pieces and RoBERTa-style Ġ-started wordpieces.

    Returns a list of:
      {
        "token": merged_word,
        "score": merged_score,
        "positions": [original_token_positions]
      }

    Notes
    -----
    - BERT continuation pieces start with "##"
    - RoBERTa/GPT-2 style new words often start with "Ġ"
    - RoBERTa continuation pieces may have no prefix at all, e.g.:
        ["Ġpro", "gramming"] -> "programming"
      so plain tokens without a word-start marker should be merged into the
      current token when one is already open.
    """
    merged: List[Dict[str, Any]] = []

    current_token = ""
    current_score = 0.0
    current_positions: List[int] = []

    def flush() -> None:
        nonlocal current_token, current_score, current_positions
        if current_token:
            merged.append({
                "token": current_token,
                "score": float(current_score),
                "positions": current_positions[:],
            })
        current_token = ""
        current_score = 0.0
        current_positions = []

    for i, (tok, score) in enumerate(zip(tokens, scores)):
        if tok in SPECIAL_TOKENS:
            flush()
            continue

        # RoBERTa / GPT-2 style explicit new-word marker
        if tok.startswith("Ġ"):
            flush()
            current_token = tok[1:]
            current_score = score
            current_positions = [i]

        # SentencePiece-style explicit new-word marker (safe extra support)
        elif tok.startswith("▁"):
            flush()
            current_token = tok[1:]
            current_score = score
            current_positions = [i]

        # BERT continuation piece
        elif tok.startswith("##"):
            piece = tok[2:]
            if current_token:
                current_token += piece
                current_score += score
                current_positions.append(i)
            else:
                current_token = piece
                current_score = score
                current_positions = [i]

        else:
            # RoBERTa continuation piece or fallback token.
            # If a token is already open, merge into it; otherwise start a new one.
            if current_token:
                current_token += tok
                current_score += score
                current_positions.append(i)
            else:
                current_token = tok
                current_score = score
                current_positions = [i]

    flush()
    return merged


def top_k_positive_merged_tokens(
    tokens: List[str],
    scores: List[float],
    k: int = 5,
    min_token_len: int = 2,
) -> List[Dict[str, Any]]:
    merged = merge_subwords(tokens, scores)

    cleaned = []
    for item in merged:
        token = item["token"].strip()

        if not token:
            continue
        if token in SPECIAL_TOKENS:
            continue
        if len(token) < min_token_len:
            continue
        if all(not ch.isalnum() for ch in token):
            continue
        if item["score"] <= 0:
            continue

        cleaned.append({
            "token": token,
            "score": float(item["score"]),
            "positions": item["positions"],
        })

    cleaned.sort(key=lambda x: x["score"], reverse=True)
    return cleaned[:k]