import re

from src.data.PolyglotDataModule import PolyglotDataModule
from src.data.WikiTextDataModule import WikiTextDataModule
from src.data.OpenAssistantDataModule import OpenAssistantDataModule
from src.data.C4DataModule import C4DataModule

# Define the base datasets dictionary
base_datasets = {
    "Polyglot": lambda directory_dataset, batch_size, shape, tokenizer_name, seed, **kwargs: PolyglotDataModule(
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        sequence_length=shape[0],
        tokenizer_name=tokenizer_name,
        seed=seed,
    ),
    "WikiText": lambda directory_dataset, batch_size, shape, tokenizer_name, seed, **kwargs: WikiTextDataModule(
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        sequence_length=shape[0],
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
    "OpenAssistant": lambda directory_dataset, batch_size, shape, tokenizer_name, seed, **kwargs: OpenAssistantDataModule(
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        sequence_length=shape[0],
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
    "C4": lambda directory_dataset, batch_size, shape, tokenizer_name, seed, **kwargs: C4DataModule(
        directory_dataset=directory_dataset,
        batch_size=batch_size,
        sequence_length=shape[0],
        tokenizer_name=tokenizer_name,
        seed=seed,
        **kwargs,
    ),
}

data_loader_map = lambda data_module: {
    "train": data_module.train_dataloader(),
    "val": data_module.val_dataloader(),
    "test": data_module.test_dataloader()
}

def get_dataset(dataset_name, directory_dataset, batch_size=1, seed=123, tokenizer_name=None, **kwargs):
    # Extract data shape from dataset name (if present)
    shape = [int(n) for n in re.findall(r"\d+", dataset_name)]
    match = re.match(r"([^0-9]+)_", dataset_name)
    if match is not None:
        dataset_name = match.group(1)

    # Get dataset
    if dataset_name in base_datasets:
        return base_datasets[dataset_name](
            directory_dataset=directory_dataset,
            batch_size=batch_size,
            shape=shape,
            tokenizer_name=tokenizer_name,
            seed=seed,
            **kwargs
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is unknown.")

