import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, sequence_length=2048, stride=512):
        self.tokenizer = tokenizer
        self.dataset=dataset
        self.texts = dataset["text"]
        tokenized_dataset = self.tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
        self.data = tokenized_dataset.input_ids[0, :-1]
        self.labels = tokenized_dataset.input_ids[0]
        self.sequence_length = sequence_length
        self.stride = stride

    def __len__(self):
        return len(self.data) // self.sequence_length

    def __getitem__(self, index):
        start_index = max(index * self.stride + self.stride - self.sequence_length, 0)
        end_index = start_index + self.stride
        if end_index > len(self.data):
            raise IndexError("Index out of bounds")
        input_ids = self.data[start_index:end_index]
        target_ids = self.labels[start_index + 1 : end_index + 1]
        target_ids[:-self.stride] = -100
        
        return input_ids, target_ids


class OpenAssistantDataModule(LightningDataModule):
    def __init__(self, directory_dataset=os.getcwd(), batch_size=64, sequence_length=2048, stride=512, n_lines=None, tokenizer_name=None, seed=1):
        super().__init__()
        self.directory_dataset = directory_dataset
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, legacy=False)
        self.sequence_length = sequence_length
        self.stride = stride
        self.n_lines = n_lines
        self.prepare_data()

    def prepare_data(self):
        # Load train, val, and test datasets
        # Load train, val, and test datasets
        train_split = "train[:95%]" if self.n_lines is None else f"train[:{self.n_lines}]"
        validation_split = "validation" if self.n_lines is None else f"validation[:{self.n_lines}]"
        test_split = "train[95%:]" if self.n_lines is None else f"train[{self.n_lines}:{2*self.n_lines}]"
        
        self.train_dataset = load_dataset("OpenAssistant/oasst1", split=train_split)
        self.val_dataset = load_dataset("OpenAssistant/oasst1", split=validation_split)
        self.test_dataset = load_dataset("OpenAssistant/oasst1", split=test_split)

    def train_dataloader(self, batch_size=None, sequence_length=None, stride=None):
        if batch_size is None:
            batch_size = self.batch_size
        if sequence_length is None:
            sequence_length = self.sequence_length
        else:
            sequence_length = min(self.sequence_length, sequence_length)
        if stride is None:
            stride = self.stride
        dataset = TextDataset(self.train_dataset, tokenizer=self.tokenizer, sequence_length=sequence_length)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader

    def val_dataloader(self, batch_size=None, sequence_length=None, stride=None):
        if batch_size is None:
            batch_size = self.batch_size
        if sequence_length is None:
            sequence_length = self.sequence_length
        else:
            sequence_length = min(self.sequence_length, sequence_length)
        if stride is None:
            stride = self.stride
        dataset = TextDataset(self.val_dataset, tokenizer=self.tokenizer, sequence_length=sequence_length)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return val_dataloader
    
    def test_dataloader(self, batch_size=None, sequence_length=None, stride=None):
        if batch_size is None:
            batch_size = self.batch_size
        if sequence_length is None:
            sequence_length = self.sequence_length
        else:
            sequence_length = min(self.sequence_length, sequence_length)
        if stride is None:
            stride = self.stride
        dataset = TextDataset(self.test_dataset, tokenizer=self.tokenizer, sequence_length=sequence_length)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return test_dataloader
