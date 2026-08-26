from src.dataset_processing.datasets.coqa import (
    CoQAEntry, ProcessedCoQAEntry, CoQAProcessor
)
from src.dataset_processing.datasets.fktc import (
    FKTCEntry, ProcessedFKTCEntry, FKTCProcessor
)
from src.dataset_processing.datasets.commonsenseqa import (
    CommonSenseQAEntry, ProcessedCommonSenseQAEntry, CommonSenseQAProcessor
)
from src.dataset_processing.datasets.triviaqa import (
    TriviaQAEntry, ProcessedTriviaQAEntry, TriviaQAProcessor
)

# Export only the main classes that users need
__all__ = [
    # CoQA
    "CoQAEntry",
    "ProcessedCoQAEntry",
    "CoQAProcessor",
    # FKTC
    "FKTCEntry",
    "ProcessedFKTCEntry",
    "FKTCProcessor",
    # CommonsenseQA
    "CommonSenseQAEntry",
    "ProcessedCommonSenseQAEntry",
    "CommonSenseQAProcessor",
    # TriviaQA
    "TriviaQAEntry",
    "ProcessedTriviaQAEntry",
    "TriviaQAProcessor"
]