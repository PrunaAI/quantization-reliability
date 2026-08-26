from datasets import load_dataset
from typing import List, Optional, Tuple

class PolyglotLoader:
    """Handles loading of raw Polyglot data."""
    
    @staticmethod
    def load_raw_data(split: str, n_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """Loads questions and answers from Polyglot dataset."""
        dataset = load_dataset("Polyglot-or-Not/Fact-Completion", split="English")
        if n_samples:
            dataset = dataset[:n_samples]
            
        questions = dataset["stem"]
        answers = dataset["true"]
        return questions, answers