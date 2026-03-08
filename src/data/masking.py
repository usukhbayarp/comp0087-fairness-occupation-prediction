import re
from typing import Iterable, List, Optional, Pattern, Sequence, Set

# Core gender terms to mask (required)
_GENDER_TERMS = [
    "he", "she",
    "his", "her",
    "him", "hers",
    "himself", "herself",
]

# Common titles / honorifics that can leak gender information
_TITLE_TERMS = [
    "mr", "mr.", "mister",
    "mrs", "mrs.", "miss", "ms", "ms.",
    "madam", "madame", "ma'am",
    "sir", "lady", "lord", "dame",
]

# Common gendered nouns. This is not exhaustive, but covers high-frequency cases.
_GENDERED_NOUN_TERMS = [
    "father", "mother", "dad", "mom", "son", "daughter",
    "brother", "sister", "husband", "wife", "boyfriend", "girlfriend",
    "uncle", "aunt", "nephew", "niece", "grandfather", "grandmother",
    "grandson", "granddaughter", "stepfather", "stepmother", "stepson", "stepdaughter",
    "businessman", "businesswoman", "chairman", "chairwoman",
    "salesman", "saleswoman", "spokesman", "spokeswoman",
    "congressman", "congresswoman", "policeman", "policewoman",
    "fireman", "stewardess", "waiter", "waitress", "actor", "actress",
    "host", "hostess", "king", "queen", "prince", "princess",
    "monk", "nun", "bride", "groom", "widow", "widower",
]

# Profession aliases used for label leakage detection.
# Keep normalized with spaces; matching is case-insensitive.
_DEFAULT_PROFESSION_ALIASES = {
    "accountant": ["accountant", "accounting professional"],
    "architect": ["architect"],
    "attorney": ["attorney", "lawyer"],
    "chiropractor": ["chiropractor"],
    "comedian": ["comedian", "comic", "stand-up comedian", "stand up comedian"],
    "composer": ["composer", "songwriter"],
    "dentist": ["dentist"],
    "dietitian": ["dietitian", "nutritionist", "dietician"],
    "dj": ["dj", "disc jockey", "deejay"],
    "filmmaker": ["filmmaker", "film maker", "director", "movie director"],
    "interior_designer": ["interior designer", "interior decorator"],
    "journalist": ["journalist", "reporter", "columnist", "correspondent"],
    "model": ["model", "fashion model"],
    "nurse": ["nurse", "registered nurse", "rn"],
    "painter": ["painter", "artist"],
    "paralegal": ["paralegal", "legal assistant"],
    "pastor": ["pastor", "minister", "priest", "reverend"],
    "personal_trainer": ["personal trainer", "fitness trainer", "trainer"],
    "photographer": ["photographer", "photojournalist"],
    "physician": ["physician", "doctor", "medical doctor"],
    "poet": ["poet", "poetess"],
    "professor": ["professor", "lecturer", "academic"],
    "psychologist": ["psychologist", "therapist", "psychotherapist"],
    "rapper": ["rapper", "hip hop artist", "mc", "emcee"],
    "software_engineer": ["software engineer", "software developer", "programmer", "developer", "engineer"],
    "surgeon": ["surgeon"],
    "teacher": ["teacher", "educator", "schoolteacher", "instructor"],
    "yoga_teacher": ["yoga teacher", "yoga instructor", "yogi"],
}


def _compile_terms_pattern(terms: Sequence[str]) -> Pattern:
    terms = sorted({t for t in terms if t}, key=len, reverse=True)
    pattern = r"\b(" + "|".join(map(re.escape, terms)) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


_GENDER_PATTERN = _compile_terms_pattern(_GENDER_TERMS)
_TITLE_PATTERN = _compile_terms_pattern(_TITLE_TERMS)
_GENDERED_NOUN_PATTERN = _compile_terms_pattern(_GENDERED_NOUN_TERMS)


def mask_gender_terms(text: str, mask_token: str = "[MASK]") -> str:
    """
    Replace explicit gendered pronouns/possessives with mask_token.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _GENDER_PATTERN.sub(mask_token, text)



def mask_titles(text: str, mask_token: str = "[MASK]") -> str:
    """Replace gendered titles / honorifics with mask_token."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _TITLE_PATTERN.sub(mask_token, text)



def mask_gendered_nouns(text: str, mask_token: str = "[MASK]") -> str:
    """Replace gendered nouns such as 'father' or 'actress' with mask_token."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _GENDERED_NOUN_PATTERN.sub(mask_token, text)



def mask_gendered_language(
    text: str,
    mask_token: str = "[MASK]",
    mask_pronouns: bool = True,
    mask_titles_flag: bool = True,
    mask_gendered_nouns_flag: bool = True,
) -> str:
    """
    Apply all selected masking rules in sequence.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    if mask_pronouns:
        text = mask_gender_terms(text, mask_token=mask_token)
    if mask_titles_flag:
        text = mask_titles(text, mask_token=mask_token)
    if mask_gendered_nouns_flag:
        text = mask_gendered_nouns(text, mask_token=mask_token)
    return text



def normalize_profession_label(label: Optional[str]) -> str:
    if label is None:
        return ""
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")



def profession_aliases_for_label(label: str, extra_aliases: Optional[Iterable[str]] = None) -> List[str]:
    """
    Return candidate surface forms for a profession label.
    Example: software_engineer -> ['software engineer', 'software_engineer', 'software-engineer', ...]
    """
    norm = normalize_profession_label(label)
    aliases: Set[str] = set()

    if norm:
        aliases.add(norm)
        aliases.add(norm.replace("_", " "))
        aliases.add(norm.replace("_", "-"))
        parts = [p for p in norm.split("_") if p]
        if parts:
            aliases.add(" ".join(parts))

    for alias in _DEFAULT_PROFESSION_ALIASES.get(norm, []):
        aliases.add(alias.strip().lower())

    if extra_aliases:
        for alias in extra_aliases:
            if alias:
                aliases.add(str(alias).strip().lower())

    return sorted(aliases, key=len, reverse=True)



def build_label_leakage_pattern(label: str, extra_aliases: Optional[Iterable[str]] = None) -> Optional[Pattern]:
    aliases = profession_aliases_for_label(label, extra_aliases=extra_aliases)
    if not aliases:
        return None
    # Replace escaped spaces with flexible whitespace so multi-word professions still match.
    escaped = []
    for alias in aliases:
        e = re.escape(alias)
        e = e.replace(r"\ ", r"\s+")
        escaped.append(e)
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)



def contains_label_leakage(text: str, label: str, extra_aliases: Optional[Iterable[str]] = None) -> bool:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    pat = build_label_leakage_pattern(label, extra_aliases=extra_aliases)
    return bool(pat.search(text)) if pat is not None else False



def mask_label_leakage(
    text: str,
    label: str,
    mask_token: str = "[LABEL_MASK]",
    extra_aliases: Optional[Iterable[str]] = None,
) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    pat = build_label_leakage_pattern(label, extra_aliases=extra_aliases)
    if pat is None:
        return text
    return pat.sub(mask_token, text)
