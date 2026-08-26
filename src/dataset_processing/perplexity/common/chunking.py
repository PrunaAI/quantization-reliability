from typing import List, Iterator
import torch

from src.dataset_processing.perplexity.common.models.dataset_entry import PerplexityDatasetEntry
from src.dataset_processing.perplexity.common.tokenization import TokenizedOutput

class DataChunker:
    """Handles chunking of tokenized data into fixed-size segments."""
    
    def __init__(self, seq_length: int, stride: int):
        self.seq_length = seq_length
        self.stride = stride
        
    def create_chunks(self, tokenized_data: TokenizedOutput) -> Iterator[PerplexityDatasetEntry]:
        """Creates chunks of data with specified sequence length and stride."""
        for i in range(0, len(tokenized_data.input_ids), self.stride):
            if i + self.seq_length >= len(tokenized_data.input_ids):
                if len(tokenized_data.input_ids) - i >= self.stride:  # Only yield if chunk is at least stride size
                    chunk = self._create_chunk(tokenized_data, i)
                    if chunk is not None:
                        yield chunk
                break
            chunk = self._create_chunk(tokenized_data, i)
            if chunk is not None:
                yield chunk
                
    def _create_chunk(self, data: TokenizedOutput, start_idx: int) -> PerplexityDatasetEntry:
        """Creates a single chunk from the data at the given start index."""
        end_idx = start_idx + self.seq_length
        
        return PerplexityDatasetEntry(
            input_ids=data.input_ids[start_idx:end_idx],
            target_ids=data.target_ids[start_idx:end_idx],
            metadata={"position": start_idx}
        )