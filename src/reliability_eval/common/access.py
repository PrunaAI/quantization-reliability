from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoreAccessPath:
    """Defines mapping between score types and their corresponding attribute paths."""
    question_aggregate: str
    sequence_aggregate: Optional[str]
    token_score: Optional[str]
