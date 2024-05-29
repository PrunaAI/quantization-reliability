import os
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset


class TextDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, sequence_length=2048):
        self.tokenizer = tokenizer
        self.questions = questions
        self.answers = answers
        self.texts = [question + " " + answer for question, answer in zip(questions, answers)]
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, index):
        tokenized_questions = self.tokenizer(
            self.questions[index], return_tensors="pt", truncation=True, max_length=self.sequence_length
        )["input_ids"][0]
        tokenized_answers = self.tokenizer(
            self.answers[index], return_tensors="pt", truncation=True, max_length=self.sequence_length
        )["input_ids"][
            0, 1:
        ]  # Remove the first token.
        data = torch.cat((tokenized_questions, tokenized_answers))
        labels = torch.cat((tokenized_questions, tokenized_answers))
        labels[: len(tokenized_questions)] = -100
        return data[:-1], labels[1:]


class PolyglotDataModule(LightningDataModule):
    def __init__(self, directory_dataset=os.getcwd(), batch_size=64, sequence_length=2048, tokenizer_name=None, seed=1):
        super().__init__()
        self.directory_dataset = directory_dataset
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, legacy=False)
        self.sequence_length = sequence_length
        self.prepare_data()

    def prepare_data(self):
        self.questions = [
            item["stem"] for item in load_dataset("Polyglot-or-Not/Fact-Completion", split="english".capitalize())
        ]
        self.answers = [
            item["true"] for item in load_dataset("Polyglot-or-Not/Fact-Completion", split="english".capitalize())
        ]

        # Create data for training, validation, and testing
        train_ratio, val_ratio = 0.8, 0.1  # assuming 10% for testing
        train_len = int(train_ratio * len(self.questions))
        val_len = int(val_ratio * len(self.questions))

        self.train_questions = self.questions[:train_len]
        self.train_answers = self.answers[:train_len]
        self.val_questions = self.questions[train_len : train_len + val_len]
        self.val_answers = self.answers[train_len : train_len + val_len]
        self.test_questions = self.questions[train_len + val_len :]
        self.test_answers = self.answers[train_len + val_len :]

    def train_dataloader(self, batch_size=None, sequence_length=None):
        if batch_size is None:
            batch_size = self.batch_size
        if sequence_length is None:
            sequence_length = self.sequence_length
        else:
            sequence_length = min(self.sequence_length, sequence_length)
        dataset = TextDataset(
            self.train_questions, self.train_answers, tokenizer=self.tokenizer, sequence_length=sequence_length
        )
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_dataloader

    def val_dataloader(self, batch_size=None, sequence_length=None):
        if batch_size is None:
            batch_size = self.batch_size
        if sequence_length is None:
            sequence_length = self.sequence_length
        else:
            sequence_length = min(self.sequence_length, sequence_length)
        dataset = TextDataset(
            self.val_questions, self.val_answers, tokenizer=self.tokenizer, sequence_length=sequence_length
        )
        val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return val_dataloader

    def test_dataloader(self, batch_size=None, sequence_length=None):
        if batch_size is None:
            batch_size = self.batch_size
        if sequence_length is None:
            sequence_length = self.sequence_length
        else:
            sequence_length = min(self.sequence_length, sequence_length)
        dataset = TextDataset(
            self.test_questions, self.test_answers, tokenizer=self.tokenizer, sequence_length=sequence_length
        )
        test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return test_dataloader
