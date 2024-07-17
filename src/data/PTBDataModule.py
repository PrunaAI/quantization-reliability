import copy
import os
import random
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, n_samples=None, text_key="sentence", sequence_length=2048, seed=0):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.texts = dataset[text_key]
        if n_samples is not None:
            self.texts = self.texts[:n_samples]
        self.n_samples = n_samples
        self.sequence_length = sequence_length
        self.seed = seed

        # Tokenize the entire dataset text
        tokenized_dataset = self.tokenizer("\n\n".join(self.texts), return_tensors="pt")
        self.data = tokenized_dataset.input_ids[0]
        self.labels = tokenized_dataset.input_ids[0]

        # Random sampling for indices
        random.seed(self.seed)
        self.indices = []
        if n_samples is not None:
            for _ in range(n_samples):
                start_idx = random.randint(0, len(self.data) - sequence_length - 1)
                self.indices.append(start_idx)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start_index = self.indices[index]
        end_index = start_index + self.sequence_length

        input_ids = self.data[start_index:end_index]
        target_ids = copy.deepcopy(self.labels[start_index + 1 : end_index + 1])
        target_ids[:-1] = -100

        return input_ids, target_ids


class PTBDataModule(LightningDataModule):
    def __init__(self, directory_dataset=os.getcwd(), batch_size=1, sequence_length=2048, n_samples=None, tokenizer_name=None, seed=1):
        super().__init__()
        self.directory_dataset = directory_dataset
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, legacy=False)
        self.sequence_length = sequence_length
        self.prepare_data()

    def prepare_data(self):
        self.train_dataset = load_dataset('ptb_text_only', 'penn_treebank', split="train", trust_remote_code=True)
        self.val_dataset = load_dataset('ptb_text_only', 'penn_treebank', split="validation", trust_remote_code=True)
        self.test_dataset = load_dataset('ptb_text_only', 'penn_treebank', split="test", trust_remote_code=True)

    def train_dataloader(self, batch_size=None, sequence_length=None, n_samples=None):
        if batch_size is None:
            batch_size = self.batch_size
        if n_samples is None:
            n_samples = self.n_samples
        if sequence_length is None:
            sequence_length = self.sequence_length
        dataset = TextDataset(self.train_dataset, tokenizer=self.tokenizer, n_samples=n_samples, sequence_length=sequence_length)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        train_dataloader.ORIGINAL_DATASET = self.train_dataset
        return train_dataloader

    def val_dataloader(self, batch_size=None, sequence_length=None, n_samples=None):
        if batch_size is None:
            batch_size = self.batch_size
        if n_samples is None:
            n_samples = self.n_samples
        if sequence_length is None:
            sequence_length = self.sequence_length
        dataset = TextDataset(self.val_dataset, tokenizer=self.tokenizer, n_samples=n_samples, sequence_length=sequence_length)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        val_dataloader.ORIGINAL_DATASET = self.val_dataset
        return val_dataloader

    def test_dataloader(self, batch_size=None, sequence_length=None, n_samples=None):
        if batch_size is None:
            batch_size = self.batch_size
        if n_samples is None:
            n_samples = self.n_samples
        if sequence_length is None:
            sequence_length = self.sequence_length
        dataset = TextDataset(self.test_dataset, tokenizer=self.tokenizer, n_samples=n_samples, sequence_length=sequence_length)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        test_dataloader.ORIGINAL_DATASET = self.test_dataset
        return test_dataloader
