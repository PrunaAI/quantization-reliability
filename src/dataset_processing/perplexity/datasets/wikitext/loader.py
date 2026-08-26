from datasets import load_dataset
from typing import List, Optional

class WikiTextLoader:
    """Handles loading of raw WikiText data."""
    
    @staticmethod
    def load_raw_data(split: str, n_samples: Optional[int] = None) -> List[str]:
        """Loads raw WikiText data from HuggingFace datasets."""
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        if n_samples is not None:
            dataset = dataset[:n_samples]
        return dataset["text"]