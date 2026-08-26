from typing import Optional
from transformers import PreTrainedTokenizer
from src.dataset_processing.perplexity.common.config.base_configs import PerplexityDatasetConfig
from src.dataset_processing.perplexity.common.enums.dataset_types import PerplexityDatasetType
from src.dataset_processing.perplexity.factory import PerplexityDatasetFactory
from src.reliability_eval.pipeline.processor.batch import DataLoader


def load_perplexity_dataset(
    dataset_name: str,
    split: str,
    tokenizer: PreTrainedTokenizer,
    n_samples: Optional[int],
    seq_length: int,
    batch_size: int,
    seed: int
) -> DataLoader:
    """Creates dataset config and loads appropriate dataset."""
    # Handle AWQ sequence length
    effective_seq_length = 512 if "awq" in tokenizer.name_or_path.lower() else seq_length
    
    config = PerplexityDatasetConfig(
        dataset_type=PerplexityDatasetType[dataset_name.upper()],
        split=split,
        n_samples=n_samples,
        seq_length=effective_seq_length,
        batch_size=batch_size,
        seed=seed,
        tokenizer_name=tokenizer.name_or_path
    )
    
    # Create processor with injected tokenizer
    processor = PerplexityDatasetFactory.create_processor(
        dataset_type=config.dataset_type,
        tokenizer=tokenizer
    )
    return processor.process_dataset(config)
