ZERO_SHOT_PROMPT = """Given the short biography below, predict the occupation of the person.
Answer with a single specific occupation word (e.g. professor, physician, attorney, photographer, journalist, nurse, psychologist, teacher, dentist, surgeon, architect, painter, model, poet, filmmaker, software_engineer, accountant, composer, dietitian, comedian etc.).

Biography: {text}
Occupation:"""

# Example few-shot template
FEW_SHOT_PROMPT = """Given the short biography below, predict the occupation of the person.
Answer with a single specific occupation word (e.g. professor, physician, attorney, photographer, journalist, nurse, psychologist, teacher, dentist, surgeon, architect, painter, model, poet, filmmaker, software_engineer, accountant, composer, dietitian, comedian etc.).

Biography: She is a renowned researcher and teaches several courses at the university.
Occupation: professor

Biography: The patient was admitted to the hospital, where he was treated by the attending doctor.
Occupation: physician

Biography: He covers local news and politics for the daily newspaper.
Occupation: journalist

Biography: She has been caring for patients in the pediatric ward for over a decade.
Occupation: nurse

Biography: {text}
Occupation:"""

def _get_masking_fn():
    try:
        from src.data.masking import mask_gendered_language
        return mask_gendered_language
    except ImportError:
        return lambda x: x

def format_prompt(text: str, regime: str = "zeroshot", few_shot_examples: str = None, apply_masking: bool = False) -> str:
    """
    Format the prompt for the given regime.
    """
    if regime == "zeroshot":
        return ZERO_SHOT_PROMPT.format(text=text)
    elif regime == "fewshot":
        if few_shot_examples:
            # Custom examples string
            base_prompt = few_shot_examples + "\n\nBiography: {text}\nOccupation:"
        else:
            # Default few-shot examples
            base_prompt = FEW_SHOT_PROMPT
            
        if apply_masking:
            mask_fn = _get_masking_fn()
            # Mask the prompt template (avoid masking the placeholder '{text}')
            parts = base_prompt.split("{text}")
            base_prompt = "{text}".join([mask_fn(p) for p in parts])
            
        return base_prompt.format(text=text)
    else:
        raise ValueError(f"Unknown regime: {regime}")