from enum import Enum


class DatasetSourceType(Enum):
    """Types of dataset sources"""
    RAW = "raw"
    PROCESSED = "processed"
    MERGED = "merged"