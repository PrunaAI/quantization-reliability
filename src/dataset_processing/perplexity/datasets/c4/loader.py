from datasets import load_dataset
from typing import List, Optional

class C4Loader:
    """Handles loading of raw C4 data."""
    
    @staticmethod
    def load_raw_data(split: str, n_samples: Optional[int] = None) -> List[str]:
        """Loads raw C4 data from HuggingFace datasets."""
        if split == "test":  # C4 doesn't have test split
            split = "validation"
            
        split_config = {
            "train": {"train": "en/c4-train.00000-of-01024.json.gz"},
            "validation": {"validation": "en/c4-validation.00000-of-00008.json.gz"}
        }
        
        dataset = load_dataset(
            "allenai/c4",
            data_files=split_config[split],
            split=split
        )
        if n_samples is not None:
            dataset = dataset[:n_samples]
        return dataset["text"]