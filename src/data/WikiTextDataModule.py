import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


# TODO: This dataset merge all independent sentences and process them as a single batch sample.
#  This is not ideal since sometimes sentences might switch from a topic to another.
#  However, it was done similarly on Wanda and SparseGPT code.
class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, sequence_length=2048):
        self.tokenizer = tokenizer
        self.dataset=dataset
        self.texts = dataset["text"]
        tokenized_dataset = self.tokenizer(" ".join(dataset["text"]), return_tensors="pt")
        self.data = tokenized_dataset.input_ids[0, :-1]
        self.labels = tokenized_dataset.input_ids[0]
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) // self.sequence_length

    def __getitem__(self, index):
        start_index = index * self.sequence_length
        end_index = (index + 1) * self.sequence_length
        return self.data[start_index:end_index], self.labels[start_index + 1 : end_index + 1]


# TODO: This dataset look at each sentence individually as a batch sample.
#  This is not ideal since sometimes sentences might not switch from a topic to another.
# class TextDataset(Dataset):
#     def __init__(self, dataset, tokenizer, sequence_length=2048):
#         self.texts = dataset["text"]
#         self.tokenizer = tokenizer
#         tokenized_dataset = self.tokenizer(
#             self.texts, return_tensors="pt", truncation=True, padding=True, max_length=sequence_length
#         )
#         self.data = tokenized_dataset.input_ids
#         self.sequence_length = sequence_length
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         return self.data[index, :-1], self.data[index, 1:]


class WikiTextDataModule(LightningDataModule):
    def __init__(self, directory_dataset=os.getcwd(), batch_size=64, sequence_length=2048, tokenizer_name=None, seed=1):
        super().__init__()
        self.directory_dataset = directory_dataset
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, legacy=False)
        self.sequence_length = sequence_length
        self.prepare_data()

    def prepare_data(self):
        # Load train, val, and test datasets
        self.train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        self.val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        self.test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    def train_dataloader(self, batch_size=None, sequence_length=None):
        if batch_size is None:
            batch_size = self.batch_size
        if sequence_length is None:
            sequence_length = self.sequence_length
        else:
            sequence_length = min(self.sequence_length, sequence_length)
        dataset = TextDataset(self.train_dataset, tokenizer=self.tokenizer, sequence_length=sequence_length)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader

    def val_dataloader(self, batch_size=None, sequence_length=None):
        if batch_size is None:
            batch_size = self.batch_size
        if sequence_length is None:
            sequence_length = self.sequence_length
        else:
            sequence_length = min(self.sequence_length, sequence_length)
        dataset = TextDataset(self.val_dataset, tokenizer=self.tokenizer, sequence_length=sequence_length)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return val_dataloader

    def test_dataloader(self, batch_size=None, sequence_length=None):
        if batch_size is None:
            batch_size = self.batch_size
        if sequence_length is None:
            sequence_length = self.sequence_length
        else:
            sequence_length = min(self.sequence_length, sequence_length)
        dataset = TextDataset(self.test_dataset, tokenizer=self.tokenizer, sequence_length=sequence_length)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return test_dataloader
