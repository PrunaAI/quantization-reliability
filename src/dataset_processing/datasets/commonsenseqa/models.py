from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CommonSenseQAEntry:
    """Single entry from CommonSenseQA dataset."""
    question: str
    choices: Dict[str, List[str]]
    answer_key: str
    question_id: str

@dataclass
class ProcessedCommonSenseQAEntry:
    """Processed entry ready for dataset creation."""
    question: str
    answer: str
    metadata: Dict[str, Any]
