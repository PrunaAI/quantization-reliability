from torch.utils.data import Dataset
from typing import List, Tuple

from src.dataset_processing.perplexity.common.models.dataset_entry import PerplexityDatasetEntry

class BasePerplexityDataset(Dataset):
    """Base dataset class for all perplexity datasets."""
    
    def __init__(self, entries: List[PerplexityDatasetEntry]):
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple:
        entry = self.entries[idx]
        return entry.input_ids, entry.target_ids