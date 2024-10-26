import re
import random
from typing import List, Optional, Tuple

from pytorch_lightning.utilities.exceptions import MisconfigurationException

from src.data.PTBDataModule import PTBDataModule
from src.data.PolyglotDataModule import PolyglotDataModule
from src.data.WikiTextDataModule import WikiTextDataModule
from src.data.OpenAssistantDataModule import OpenAssistantDataModule
from src.data.C4DataModule import C4DataModule

from src.data.CoQA import load_coqa_dataset_pairs
from src.data.FKTC_datasets import load_fktc
from src.data.ToyQADataset import toy_qa_dataset

from src.data.constants import COQA_PATH, DATA_FILES
from src.reliability.apply_typos import apply_typo_modifications

import logging

from src.reliability.create_typos_list import create_typo_dict
logger = logging.getLogger("quant_logger")


# Define the base datasets dictionary
base_datasets = {
    "Polyglot": lambda batch_size, sequence_length, stride, tokenizer_name, seed, **kwargs: PolyglotDataModule(
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride=stride,
        tokenizer_name=tokenizer_name,
        seed=seed,
    ),
    "WikiText": lambda batch_size, sequence_length, stride, tokenizer_name, seed, **kwargs: WikiTextDataModule(
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride=stride,
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
    "OpenAssistant": lambda batch_size, sequence_length, stride, tokenizer_name, seed, **kwargs: OpenAssistantDataModule(
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride=stride,
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
    "C4": lambda batch_size, sequence_length, stride, tokenizer_name, seed, **kwargs: C4DataModule(
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride=stride,
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
    "PTB": lambda batch_size, sequence_length, stride, tokenizer_name, seed, **kwargs: PTBDataModule(
        batch_size=batch_size,
        sequence_length=sequence_length,
        stride=stride,
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
}

def data_loader_from_split(data_module, split, batch_size=None, sequence_length=None, stride=None):
    if split == "train":
        return data_module.train_dataloader(batch_size=batch_size, sequence_length=sequence_length, stride=stride)
    elif split == "validation":
        return data_module.val_dataloader(batch_size=batch_size, sequence_length=sequence_length, stride=stride)
    elif split == "test":
        return data_module.test_dataloader(batch_size=batch_size, sequence_length=sequence_length, stride=stride)
    else:
        raise MisconfigurationException(f"Split {split} is not valid. Must be one of ['train', 'validation', 'test']")

def get_dataset(dataset_name, batch_size=1, sequence_length=2048, stride=512, seed=123, tokenizer_name=None, **kwargs):
    # Get dataset
    if dataset_name in base_datasets:
        logger.info(f"Loading dataset {dataset_name} with the following configuration:")
        logger.info(f"  batch_size: {batch_size}")
        logger.info(f"  sequence_length: {sequence_length}")
        logger.info(f"  stride: {stride}")
        logger.info(f"  seed: {seed}")
        logger.info(f"  tokenizer_name: {tokenizer_name}")
        return base_datasets[dataset_name](
            batch_size=batch_size,
            sequence_length=sequence_length,
            stride=stride,
            tokenizer_name=tokenizer_name,
            seed=seed,
            **kwargs
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is unknown.")

    
def load_dataset_from_name(
    dataset_name: str,
    max_entries: Optional[int] = None,
    typo_type: str = "none",
    typo_intensity: int = 0,
    max_relations: int = 1
) -> List[Tuple[str, str]]:
    """
    High-level function to load different datasets (FKTC, CoQA, toy-qa-dataset).
    
    Args:
        dataset_name: Name of the dataset to load ('coqa', specific FKTC dataset, or 'toy-qa-dataset')
        max_entries: Maximum number of entries to load (None for all)
        typo_type: Type of typo to apply ("none" for no typos)
        typo_intensity: Intensity of typo modifications
        max_relations: Maximum number of relations to use (only for FKTC datasets)
        
    Returns:
        List of (question, answer) tuples
    """
    if dataset_name == "toy-qa-dataset":
        dataset = toy_qa_dataset
        if typo_type != "none":
            typo_dict = create_typo_dict(typo_type, typo_intensity)
            dataset = [(apply_typo_modifications(q, typo_dict, [a]), a) for q, a in dataset]
        return dataset
    
    elif dataset_name == "coqa":
        return load_coqa_dataset_pairs(
            file_path=COQA_PATH,
            max_entries=max_entries,
            typo_type=typo_type,
            typo_intensity=typo_intensity
        )
    
    elif dataset_name in DATA_FILES:
        return load_fktc(
            dataset_name=dataset_name,
            max_relations=max_relations,
            max_entries=max_entries,
            typo_type=typo_type,
            typo_intensity=typo_intensity
        )
    
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}. Must be 'coqa', 'toy-qa-dataset', or one of {DATA_FILES}")

