def create_typo_dict(typo_type: str, intensity: int) -> Dict[str, int]:
    """
    Creates a dictionary of typo modifications with specified intensity.
    Reuses the same typo types as the original implementation.
    """
    base_dict = {
        "char_insertion": 0,
        "char_deletion": 0,
        "char_replacement": 0,
        "char_repetition": 0,
        "char_swapping": 0,
        "word_CMW": 0,
        "char_LCC": 0,
        "word_synonym": 0,
        "char_insert_noise": 0,
        "word_repeat": 0,
        "char_substitution": 0,
        "word_emoji": 0,
        "word_internet_slang": 0,
        "word_phrase_translation": 0,
        "word_context_aware_insertion": 0,
        "word_remove_punctuation": 0,
        "word_keyword_only": 0,
        "word_taxonomy_pos": 0,
        "word_taxonomy_neg": 0
    }
    
    if typo_type in base_dict:
        base_dict[typo_type] = intensity
    elif typo_type == "random":
        for _ in range(intensity):
            key = random.choice(list(base_dict.keys()))
            base_dict[key] += 1
    
    return base_dict