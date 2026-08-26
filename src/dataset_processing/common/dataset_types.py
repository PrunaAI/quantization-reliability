from enum import Enum, auto


class DatasetType(Enum):
    """Types of supported datasets"""
    FKTC = "FKTC"
    COQA = "COQA"
    TRIVIAQA = "TRIVIAQA"
    COMMONSENSEQA = "COMMONSENSEQA"
    MMLU = "MMLU"
    TOY = "TOY"