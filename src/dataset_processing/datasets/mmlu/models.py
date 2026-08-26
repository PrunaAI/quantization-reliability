from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class MMLUEntry:
    """Single entry from MMLU dataset."""
    question: str
    choices: List[str]
    answer_key: str
    subject: str
    question_id: str

@dataclass
class ProcessedMMLUEntry:
    """Processed entry ready for dataset creation."""
    question: str
    answer: str
    metadata: Dict[str, Any]