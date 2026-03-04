# masking.py
import re
from typing import Pattern

# Core gender terms to mask (required)
_GENDER_TERMS = [
    "he", "she",
    "his", "her",
    "him", "hers",
    "himself", "herself",
]

def _build_gender_pattern() -> Pattern:
    # word boundary to avoid masking substrings (e.g., "the")
    # case-insensitive
    terms = sorted(_GENDER_TERMS, key=len, reverse=True)
    pattern = r"\b(" + "|".join(map(re.escape, terms)) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)

_GENDER_PATTERN = _build_gender_pattern()

def mask_gender_terms(text: str, mask_token: str = "[MASK]") -> str:
    """
    Replace explicit gendered terms with mask_token.

    Example:
      "She is a doctor. Her research..." -> "[MASK] is a doctor. [MASK] research..."
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _GENDER_PATTERN.sub(mask_token, text)