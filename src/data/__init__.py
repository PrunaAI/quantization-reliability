import re

from pytorch_lightning.utilities.exceptions import MisconfigurationException

from src.data.PTBDataModule import PTBDataModule
from src.data.PolyglotDataModule import PolyglotDataModule
from src.data.WikiTextDataModule import WikiTextDataModule
from src.data.OpenAssistantDataModule import OpenAssistantDataModule
from src.data.C4DataModule import C4DataModule

import logging
logger = logging.getLogger("quant_logger")

# Define the base datasets dictionary
base_datasets = {
    "Polyglot": lambda directory_dataset, batch_size, n_samples, sequence_length, tokenizer_name, seed, **kwargs: PolyglotDataModule(
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        n_samples=n_samples,
        sequence_length=sequence_length,
        tokenizer_name=tokenizer_name,
        seed=seed,
    ),
    "WikiText": lambda directory_dataset, batch_size, n_samples, sequence_length, tokenizer_name, seed, **kwargs: WikiTextDataModule(
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        n_samples=n_samples,
        sequence_length=sequence_length,
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
    "OpenAssistant": lambda directory_dataset, batch_size, n_samples, sequence_length, tokenizer_name, seed, **kwargs: OpenAssistantDataModule(
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        n_samples=n_samples,
        sequence_length=sequence_length,
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
    "C4": lambda directory_dataset, batch_size, n_samples, sequence_length, tokenizer_name, seed, **kwargs: C4DataModule(
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        n_samples=n_samples,
        sequence_length=sequence_length,
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
    "PTB": lambda directory_dataset, batch_size, n_samples, sequence_length, tokenizer_name, seed, **kwargs: PTBDataModule(
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        n_samples=n_samples,
        sequence_length=sequence_length,
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
}

def data_loader_from_split(data_module, split, batch_size=None, n_samples=None, sequence_length=None):
    if split == "train":
        return data_module.train_dataloader(batch_size=batch_size, n_samples=n_samples, sequence_length=sequence_length)
    elif split == "validation":
        return data_module.val_dataloader(batch_size=batch_size, n_samples=n_samples, sequence_length=sequence_length)
    elif split == "test":
        return data_module.test_dataloader(batch_size=batch_size, n_samples=n_samples, sequence_length=sequence_length)
    else:
        raise MisconfigurationException(f"Split {split} is not valid. Must be one of ['train', 'validation', 'test']")

def get_dataset(dataset_name, directory_dataset, batch_size=1, n_samples=128, sequence_length=512, seed=123, tokenizer_name=None, **kwargs):
    # Get dataset
    if dataset_name in base_datasets:
        logger.info(f"Loading dataset {dataset_name} with the following configuration:")
        logger.info(f"  directory_dataset: {directory_dataset}")
        logger.info(f"  batch_size: {batch_size}")
        logger.info(f"  n_samples: {n_samples}")
        logger.info(f"  sequence_length: {sequence_length}")
        logger.info(f"  seed: {seed}")
        logger.info(f"  tokenizer_name: {tokenizer_name}")
        return base_datasets[dataset_name](
            directory_dataset=directory_dataset,
            batch_size=batch_size,
            n_samples=n_samples,
            sequence_length=sequence_length,
            tokenizer_name=tokenizer_name,
            seed=seed,
            **kwargs
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is unknown.")

