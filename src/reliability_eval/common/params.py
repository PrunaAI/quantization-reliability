from enum import Enum


class ExperimentConfigParam(str, Enum):
    """High-level experiment configuration parameters."""
    GENERATION_STRATEGY = "generation_strategy"
    DATASET_NAME = "dataset_name"
    NUM_SHOTS = "num_shots"
    PROMPT_STRATEGY = "prompt_strategy"
    NUM_REPEATS = "num_repeats"
    TOP_K_CONCENTRATION = "top_k_concentration"

class GenerationConfigParam(str, Enum):
    """Low-level model generation parameters."""
    MAX_NEW_TOKENS = "max_new_tokens"
    DO_SAMPLE = "do_sample"
    USE_CACHE = "use_cache"
    LENGTH_PENALTY = "length_penalty"
    OUTPUT_SCORES = "output_scores"
    OUTPUT_HIDDEN_STATES = "output_hidden_states"
    OUTPUT_ATTENTIONS = "output_attentions"
    OUTPUT_LOGITS = "output_logits"
    RETURN_DICT_IN_GENERATE = "return_dict_in_generate"
    PAD_TOKEN_ID = "pad_token_id"
    NUM_BEAMS = "num_beams"
    NUM_RETURN_SEQUENCES = "num_return_sequences"
    NUM_BEAM_GROUPS = "num_beam_groups"
    PENALTY_ALPHA = "penalty_alpha"
    TOP_K = "top_k"
    TOP_P = "top_p"
    TEMPERATURE = "temperature"
