ZERO_SHOT_PROMPT = """Given the short biography below, predict the occupation of the person.
Answer with a single specific occupation word (e.g., professor, nurse, physician, journalist, etc.).

Biography: {text}
Occupation:"""

# Example few-shot template
FEW_SHOT_PROMPT = """Given the short biography below, predict the occupation of the person.
Answer with a single specific occupation word (e.g. professor, nurse, physician, journalist, etc.).

Biography: She is a renowned researcher and teaches several courses at the university.
Occupation: professor

Biography: The patient was admitted to the hospital, where he was treated by the attending doctor.
Occupation: physician

Biography: He covers local news and politics for the daily newspaper.
Occupation: journalist

Biography: {text}
Occupation:"""

def format_prompt(text: str, regime: str = "zeroshot", few_shot_examples: str = None) -> str:
    """
    Format the prompt for the given regime.
    """
    if regime == "zeroshot":
        return ZERO_SHOT_PROMPT.format(text=text)
    elif regime == "fewshot":
        if few_shot_examples:
            # Custom examples string
            return few_shot_examples + f"\n\nBiography: {text}\nOccupation:"
        else:
            # Default few-shot examples
            return FEW_SHOT_PROMPT.format(text=text)
    else:
        raise ValueError(f"Unknown regime: {regime}")