from enum import Enum


class GenerationStrategy(Enum):
    GREEDY_SEARCH = "greedy_search"  # Baseline - use num_rep=1
    CONTRASTIVE_SEARCH = "contrastive_search"
    MULTINOMIAL_SAMPLING = "multinomial_sampling"  # Baseline
    TOP_K_SAMPLING = "top_k_sampling"
    TOP_P_SAMPLING = "top_p_sampling"
    BEAM_SEARCH = "beam_search"   # Baseline - use num_rep=1 - try it if it makes sense!
    BEAM_SEARCH_WITH_SAMPLING = "beam_search_with_sampling"
    DIVERSE_BEAM_SEARCH = "diverse_beam_search"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    ASSISTED_DECODING = "assisted_decoding"
    DOLA = "dola"

