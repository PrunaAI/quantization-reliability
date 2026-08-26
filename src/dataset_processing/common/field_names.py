from enum import Enum


class DatasetEntryNames(Enum):
    """Names of dataset entry fields"""
    QUESTION = "question"
    ANSWER = "answer"
    METADATA = "metadata"
    
class PerturbationNames(Enum):
    """Names of perturbation-related fields"""
    PERTURBATION_TYPE = "perturbation_type"
    PERTURBATION_INTENSITY = "perturbation_intensity"