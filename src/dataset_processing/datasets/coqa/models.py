from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CoQAEntry:
    """Single entry from CoQA dataset."""
    story: str
    question: str
    answer: str
    question_id: int
    story_id: str

@dataclass
class ProcessedCoQAEntry:
    """Processed entry ready for dataset creation."""
    question: str
    answer: str
    metadata: Dict[str, Any]
