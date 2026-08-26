from enum import Enum, auto

class PERTURBATION_CATEGORY(Enum):
    """Enumeration of perturbation categories."""
    CHARACTER = auto()
    WORD = auto()

class PerturbationType(Enum):
    """Enumeration of all available perturbation types."""
    NONE = "none"
    
    # Character-level perturbations
    CHAR_INSERTION = "char_insertion"
    CHAR_DELETION = "char_deletion"
    CHAR_REPLACEMENT = "char_replacement"
    CHAR_SWAPPING = "char_swapping"
    CHAR_REPETITION = "char_repetition"
    CHAR_SUBSTITUTION = "char_substitution"
    CHAR_INSERT_NOISE = "char_insert_noise"
    CHAR_CASE_CHANGE = "char_LCC"
    CHAR_EMOJI = "char_emoji"
    
    # Word-level perturbations
    WORD_INSERTION = "word_context_aware_insertion"
    WORD_DELETION = "word_keyword_only"
    WORD_SYNONYM = "word_synonym_replacement"
    WORD_SWAPPING = "word_swapping"
    WORD_REPETITION = "word_repeat"
    WORD_INTERNET_SLANG = "word_internet_slang"
    WORD_PHRASE_TRANSLATION = "word_phrase_translation"
    
    # Taxonomy perturbations
    WORD_TAXONOMY_POS = "word_taxonomy_pos"
    WORD_TAXONOMY_NEG = "word_taxonomy_neg"
    
    @classmethod
    def get_all_perturbations(cls) -> list:
        """Returns all perturbation types except NONE."""
        return [t for t in cls if t != cls.NONE]
    
    @classmethod
    def get_char_perturbations(cls) -> list:
        """Returns all character-level perturbations."""
        return [t for t in cls if t.value.startswith("char_")]
    
    @classmethod
    def get_word_perturbations(cls) -> list:
        """Returns all word-level perturbations."""
        return [t for t in cls if t.value.startswith("word_")]
    
    def __str__(self):
        return self.value
