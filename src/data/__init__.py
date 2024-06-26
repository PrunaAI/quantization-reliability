import re

from pytorch_lightning.utilities.exceptions import MisconfigurationException

from src.data.PolyglotDataModule import PolyglotDataModule
from src.data.WikiTextDataModule import WikiTextDataModule
from src.data.OpenAssistantDataModule import OpenAssistantDataModule
from src.data.C4DataModule import C4DataModule

# Define the base datasets dictionary
base_datasets = {
    "Polyglot": lambda directory_dataset, batch_size, sequence_length, tokenizer_name, seed, **kwargs: PolyglotDataModule(
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        sequence_length=sequence_length,
        tokenizer_name=tokenizer_name,
        seed=seed,
    ),
    "WikiText": lambda directory_dataset, batch_size, sequence_length, tokenizer_name, seed, **kwargs: WikiTextDataModule(
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        sequence_length=sequence_length,
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
    "OpenAssistant": lambda directory_dataset, batch_size, sequence_length, tokenizer_name, seed, **kwargs: OpenAssistantDataModule(
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        sequence_length=sequence_length,
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
    "C4": lambda directory_dataset, batch_size, sequence_length, tokenizer_name, seed, **kwargs: C4DataModule(
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        sequence_length=sequence_length,
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
}

def get_data_loader_from_split(data_module, split):
    if split not in data_module.splits:
        print(f"Split {split} is not in {data_module.splits} for data module {data_module}. Returning validation data loader.")
        split = "validation"
        
    if split == "train":
        return data_module.train_dataloader()
    if split == "validation":
        return data_module.val_dataloader()
    return data_module.test_dataloader()

def get_dataset(dataset_name, directory_dataset, batch_size=1, sequence_length=512, seed=123, tokenizer_name=None, **kwargs):
    # Get dataset
    if dataset_name in base_datasets:
        return base_datasets[dataset_name](
            directory_dataset=directory_dataset,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tokenizer_name=tokenizer_name,
            seed=seed,
            **kwargs
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is unknown.")

