from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class FKTCEntry:
    """Single entry from FKTC dataset"""
    subject: str
    object: str
    taxonomy: List[str]
    relations: List[str]

@dataclass
class ProcessedFKTCEntry:
    """Processed entry ready for dataset creation"""
    question: str
    answer: str
    metadata: Dict[str, Any]
