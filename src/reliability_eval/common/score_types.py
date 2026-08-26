# (batch_size, num_beams * num_repeats, max_new_tokens)
# sequence aggregation: aggregate over the same query

# output shape (batch_size, num_repeats, max_new_tokens)
from enum import Enum


class TokenScoreTypes(Enum):
    TOKEN_INFO_TUPLES = "token_info_tuples"
    RELEVANT_DECODED_TEXT = "relevant_decoded_text"
    FULL_DECODED_TEXT = "full_decoded_text"
    EFFECTIVE_LENGTHS = "effective_lengths"
    PADDED_SEQUENCES = "padded_sequences"
    CROSS_ENTROPY = "cross_entropy"
    NLL = "nll"
    ENTROPY = "entropy"
    TOP_K = "top_k"
    SEMANTIC_ENTROPY = "semantic_entropy"

# output shape (batch_size, num_repeats, 1)
class SequenceAggregateTypes(Enum):
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"

# output shape (B, S, 1)
class SequenceScoreTypes(Enum):
    ID = "id"
    IS_CORRECT = "is_correct"

# output shape (B, 1, 1)
class QuestionAggregateTypes(Enum):
    ACCURACY = "accuracy"
    MEAN = "mean"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    MEAN_AMONG_CORRECT = "mean_among_correct"

# output shape (1, 1, 1)
class DatasetMetricTypes(Enum):
    ACCURACY = "accuracy"
    AUCROC = "aucroc"
    AUCPR = "aucpr"
    BRIER = "brier"
    MEAN_SCORES = "mean_scores"
