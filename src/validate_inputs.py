import logging
from typing import Optional
from src.models import BASE_MODELS, HF_QUANTIZED_MODELS, LOCAL_QUANTIZED_MODELS
from src.data import base_datasets, DATA_FILES

logger_name = "quant_logger"
logger = logging.getLogger(logger_name)

def validate_run_evaluate_inputs(
    exp_id: str,
    model_name: str,
    dataset_name: str,
    typo_type: str,
    typo_intensity: int,
    strategy: str,
    max_new_tokens: int,
    temperature: float,
    use_beam_search: bool,
    n_repeats: int,
    n_beams: int,
    max_relations: Optional[int],
    max_entries: Optional[int]
) -> None:
    """
    Validates inputs for the run_evaluate function.
    Raises ValueError with detailed message if validation fails.
    """
    # Valid model names (combining all model dictionaries)
    VALID_MODELS = set(list(BASE_MODELS.keys()) + 
                      list(HF_QUANTIZED_MODELS.keys()) + 
                      list(LOCAL_QUANTIZED_MODELS.keys()))

    # Valid datasets (combining all dataset sources)
    VALID_DATASETS = set(list(base_datasets.keys()) + 
                        list(DATA_FILES) + 
                        ['coqa', 'toy-qa-dataset'])

    # Valid strategies based on usage in the codebase
    VALID_STRATEGIES = {
        "Direct Completion",
        "Step by Step",
        "Few Shot",
        "Chain of Thought",
        "Tree of Thoughts"
    }

    # Valid typo types based on the typo modifications in apply_typo_modifications
    VALID_TYPO_TYPES = {
        "none",
        "word_CMW",
        "word_synonym",
        "char_insert_noise",
        "char_substitution",
        "char_insertion",
        "char_deletion",
        "char_replacement",
        "char_repetition",
        "char_swapping",
        "char_LCC",
        "word_emoji",
        "word_internet_slang",
        "word_phrase_translation",
        "word_repeat",
        "word_context_aware_insertion",
        "word_remove_punctuation",
        "word_keyword_only",
        "word_taxonomy_pos",
        "word_taxonomy_neg"
    }

    # Validate exp_id
    if not exp_id or not isinstance(exp_id, str):
        raise ValueError("exp_id must be a non-empty string")

    # Validate model_name
    if model_name not in VALID_MODELS:
        raise ValueError(f"Invalid model_name: {model_name}. Must be one of: {sorted(VALID_MODELS)}")

    # Validate dataset_name
    if dataset_name not in VALID_DATASETS:
        raise ValueError(f"Invalid dataset_name: {dataset_name}. Must be one of: {sorted(VALID_DATASETS)}")

    # Validate typo_type
    if typo_type not in VALID_TYPO_TYPES:
        raise ValueError(f"Invalid typo_type: {typo_type}. Must be one of: {sorted(VALID_TYPO_TYPES)}")

    # Validate typo_intensity
    if not isinstance(typo_intensity, int) or typo_intensity < 0:
        raise ValueError("typo_intensity must be a non-negative integer")

    # Validate strategy
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Invalid strategy: {strategy}. Must be one of: {sorted(VALID_STRATEGIES)}")

    # Validate max_new_tokens
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")

    # Validate temperature
    if not isinstance(temperature, (int, float)) or temperature < 0:
        raise ValueError("temperature must be a non-negative number")

    # Validate use_beam_search
    if not isinstance(use_beam_search, bool):
        raise ValueError("use_beam_search must be a boolean")

    # Validate n_repeats
    if not isinstance(n_repeats, int) or n_repeats <= 0:
        raise ValueError("n_repeats must be a positive integer")

    # Validate n_beams
    if not isinstance(n_beams, int) or n_beams <= 0:
        raise ValueError("n_beams must be a positive integer")

    # Validate max_relations
    if max_relations is not None and (not isinstance(max_relations, int) or max_relations <= 0):
        raise ValueError("max_relations must be None or a positive integer")

    # Validate max_entries
    if max_entries is not None and (not isinstance(max_entries, int) or max_entries <= 0):
        raise ValueError("max_entries must be None or a positive integer")

    # Log successful validation
    logger.info("Input validation completed successfully")
    
    return True