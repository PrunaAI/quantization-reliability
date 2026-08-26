from datasets import load_dataset
from typing import List, Optional

class OpenAssistantLoader:
    """Handles loading of raw OpenAssistant data."""
    
    @staticmethod
    def load_raw_data(split: str, n_samples: Optional[int] = None) -> List[str]:
        """Loads raw OpenAssistant data and concatenates conversations."""
        dataset = load_dataset("OpenAssistant/oasst1", split=split)
        if n_samples:
            dataset = dataset[:n_samples]
        return [entry for entry in dataset["text"]]