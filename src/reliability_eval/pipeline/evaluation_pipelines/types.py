from enum import Enum

class PipelineType(Enum):
    """Available pipeline types for reliability evaluation."""
    NLL = "nll_pipeline"
    CONFIDENCE = "confidence_pipeline"
    ENTROPY = "entropy_pipeline"
    TOPK = "topk_pipeline"
    ACCURACY = "accuracy_pipeline"
    SEMANTIC_ENTROPY = "semantic_entropy_pipeline"
