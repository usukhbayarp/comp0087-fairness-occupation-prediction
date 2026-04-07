from __future__ import annotations

from typing import List, Dict, Any


SPECIAL_TOKENS = {
    "[CLS]", "[SEP]", "[PAD]", "[MASK]",
    "<s>", "</s>", "<pad>", "<mask>"
}


def merge_subwords(tokens: List[str], scores: List[float]) -> List[Dict[str, Any]]:
    """
    Merge BERT-style ## pieces and RoBERTa/SentencePiece-style wordpieces.

    Rules:
    - BERT/DistilBERT:
        - plain tokens start new words
        - ## continues previous word
    - RoBERTa/GPT2:
        - Ġ marks a new word
        - plain tokens without Ġ are continuations of the current word
    - SentencePiece:
        - ▁ marks a new word
        - plain tokens without ▁ are continuations of the current word
    """
    merged: List[Dict[str, Any]] = []

    current_token = ""
    current_score = 0.0
    current_positions: List[int] = []

    # Detect tokenizer family heuristically from the token stream
    has_g_marker = any(tok.startswith("Ġ") for tok in tokens)
    has_sp_marker = any(tok.startswith("▁") for tok in tokens)

    # If we see Ġ or ▁ anywhere, plain non-prefixed tokens are treated as continuations.
    roberta_like = has_g_marker or has_sp_marker

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

        # RoBERTa / GPT-2 explicit new-word marker
        if tok.startswith("Ġ"):
            flush()
            current_token = tok[1:]
            current_score = score
            current_positions = [i]

        # SentencePiece explicit new-word marker
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
            if roberta_like:
                # In RoBERTa-like tokenization, plain tokens are continuations
                if current_token:
                    current_token += tok
                    current_score += score
                    current_positions.append(i)
                else:
                    current_token = tok
                    current_score = score
                    current_positions = [i]
            else:
                # In BERT-like tokenization, plain tokens start new words
                flush()
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