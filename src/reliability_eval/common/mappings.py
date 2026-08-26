from src.dataset_processing.common.dataset_types import DatasetType
from src.dataset_processing.common.source_types import DatasetSourceType
from src.dataset_processing.perturbations.enums import PerturbationType
from src.reliability_eval.common.generation import GenerationStrategy
from src.reliability_eval.pipeline.evaluation_pipelines.types import PipelineType
from src.reliability_eval.prompting.types import PromptStrategy


DATASET_TYPE_MAP = {
    "FKTC": DatasetType.FKTC,
    "COQA": DatasetType.COQA,
    "TRIVIAQA": DatasetType.TRIVIAQA,
    "COMMONSENSEQA": DatasetType.COMMONSENSEQA,
    "MMLU": DatasetType.MMLU
}

SOURCE_TYPE_MAP = {
    "raw": DatasetSourceType.RAW,
    "processed": DatasetSourceType.PROCESSED,
    "merged": DatasetSourceType.MERGED
}

PERTURBATION_TYPE_MAP = {
    "none": PerturbationType.NONE,
    "char_insertion": PerturbationType.CHAR_INSERTION,
    "char_deletion": PerturbationType.CHAR_DELETION,
    "char_replacement": PerturbationType.CHAR_REPLACEMENT,
    "char_swapping": PerturbationType.CHAR_SWAPPING,
    "char_repetition": PerturbationType.CHAR_REPETITION,
    "char_substitution": PerturbationType.CHAR_SUBSTITUTION,
    "char_insert_noise": PerturbationType.CHAR_INSERT_NOISE,
    "char_LCC": PerturbationType.CHAR_CASE_CHANGE,
    "char_emoji": PerturbationType.CHAR_EMOJI,
    "word_context_aware_insertion": PerturbationType.WORD_INSERTION,
    "word_keyword_only": PerturbationType.WORD_DELETION,
    "word_synonym_replacement": PerturbationType.WORD_SYNONYM,
    "word_swapping": PerturbationType.WORD_SWAPPING,
    "word_repeat": PerturbationType.WORD_REPETITION,
    "word_internet_slang": PerturbationType.WORD_INTERNET_SLANG,
    "word_phrase_translation": PerturbationType.WORD_PHRASE_TRANSLATION,
    "word_taxonomy_pos": PerturbationType.WORD_TAXONOMY_POS,
    "word_taxonomy_neg": PerturbationType.WORD_TAXONOMY_NEG
}

PROMPT_STRATEGY_MAP = {
    "Original": PromptStrategy.ORIGINAL,
    "Fact Statement": PromptStrategy.FACT_STATEMENT,
    "Completion": PromptStrategy.COMPLETION,
    "Definitive Statement": PromptStrategy.DEFINITIVE_STATEMENT,
    "Fill-in-the-Blank": PromptStrategy.FILL_IN_BLANK,
    "Structured Answer": PromptStrategy.STRUCTURED_ANSWER,
    "Direct Instruction": PromptStrategy.DIRECT_INSTRUCTION,
    "Contextual": PromptStrategy.CONTEXTUAL,
    "Question-Answer Pairs": PromptStrategy.QUESTION_ANSWER_PAIRS,
    "Direct Answer": PromptStrategy.DIRECT_ANSWER,
    "Q&A Format": PromptStrategy.QA_FORMAT,
    "Instructional": PromptStrategy.INSTRUCTIONAL,
    "Summary": PromptStrategy.SUMMARY,
    "Echo": PromptStrategy.ECHO,
    "True Completion": PromptStrategy.TRUE_COMPLETION,
    "Direct Completion": PromptStrategy.DIRECT_COMPLETION,
    "Answer Completion": PromptStrategy.ANSWER_COMPLETION,
    "Direct Query": PromptStrategy.DIRECT_QUERY,
    "Factual Retrieval": PromptStrategy.FACTUAL_RETRIEVAL,
    "First Thought": PromptStrategy.FIRST_THOUGHT,
    "Deductive Reasoning": PromptStrategy.DEDUCTIVE_REASONING,
    "Expert Persona": PromptStrategy.EXPERT_PERSONA,
    "Reflective Reasoning": PromptStrategy.REFLECTIVE_REASONING,
    "Zero-Shot": PromptStrategy.ZERO_SHOT,
    "One-Shot": PromptStrategy.ONE_SHOT,
    "Two-Shot": PromptStrategy.TWO_SHOT,
    "Three-Shot": PromptStrategy.THREE_SHOT,
    "Four-Shot": PromptStrategy.FOUR_SHOT,
    "Five-Shot": PromptStrategy.FIVE_SHOT
}

GENERATION_STRATEGY_MAP = {
    "greedy_search": GenerationStrategy.GREEDY_SEARCH,
    "contrastive_search": GenerationStrategy.CONTRASTIVE_SEARCH,
    "multinomial_sampling": GenerationStrategy.MULTINOMIAL_SAMPLING,
    "top_k_sampling": GenerationStrategy.TOP_K_SAMPLING,
    "top_p_sampling": GenerationStrategy.TOP_P_SAMPLING,
    "beam_search": GenerationStrategy.BEAM_SEARCH,
    "beam_search_with_sampling": GenerationStrategy.BEAM_SEARCH_WITH_SAMPLING,
    "diverse_beam_search": GenerationStrategy.DIVERSE_BEAM_SEARCH,
    "constrained_beam_search": GenerationStrategy.CONSTRAINED_BEAM_SEARCH,
    "assisted_decoding": GenerationStrategy.ASSISTED_DECODING,
    "dola": GenerationStrategy.DOLA
}

PIPELINE_TYPE_MAP = {
    "nll_pipeline": PipelineType.NLL,
    "confidence_pipeline": PipelineType.CONFIDENCE,
    "entropy_pipeline": PipelineType.ENTROPY,
    "topk_pipeline": PipelineType.TOPK,
    "accuracy_pipeline": PipelineType.ACCURACY,
    "semantic_entropy_pipeline": PipelineType.SEMANTIC_ENTROPY
}
