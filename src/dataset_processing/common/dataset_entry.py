from typing import Any, Dict, List, Union


class DatasetEntry:
    """Base class for dataset entries"""
    def __init__(self, question: str, answer: Union[str, List[str]], metadata: Dict[str, Any] = None):
        self.question = question
        self.answer = answer
        self.metadata = metadata or {}