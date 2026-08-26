from datasets import load_dataset
from typing import List, Optional

class PTBLoader:
    """Handles loading of raw PTB data."""
    
    @staticmethod
    def load_raw_data(split: str, n_samples: Optional[int] = None) -> List[str]:
        """Loads raw PTB data from HuggingFace datasets."""
        dataset = load_dataset('ptb_text_only', 'penn_treebank', split=split, trust_remote_code=True)
        if n_samples is not None:
            dataset = dataset[:n_samples]
            
        # Convert 'sentence' field to 'text' for consistency
        dataset = {"text": dataset.pop("sentence")}
        return dataset["text"]