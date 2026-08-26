from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TriviaQAEntry:
    """Single entry from TriviaQA dataset."""
    question: str
    answer_list: List[str]
    question_id: str

@dataclass
class ProcessedTriviaQAEntry:
    """Processed entry ready for dataset creation."""
    question: str
    answer: str
    metadata: Dict[str, Any]
