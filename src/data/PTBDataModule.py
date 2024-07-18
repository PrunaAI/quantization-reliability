import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

# Function to rename a feature key
def rename_feature(dataset, old_key, new_key):
    return dataset.map(lambda example: {new_key: example[old_key]}, remove_columns=[old_key])


class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, text_key="sentence", sequence_length=2048, stride=512, seed=0):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.texts = dataset[text_key]
        self.sequence_length = sequence_length
        self.stride = stride
        self.seed = seed

        # Tokenize the entire dataset text
        tokenized_dataset = self.tokenizer("\n\n".join(self.texts), return_tensors="pt")
        self.data = tokenized_dataset.input_ids[0, :-1]
        self.labels = tokenized_dataset.input_ids[0]

    def __len__(self):
        return len(self.data) // self.stride

    def __getitem__(self, index):        
        start_index = max(index * self.stride + self.stride - self.sequence_length, 0)
        end_index = start_index + self.stride
        if end_index > len(self.data):
            raise IndexError("Index out of bounds")
        input_ids = self.data[start_index:end_index]
        target_ids = self.labels[start_index + 1: end_index + 1]
        target_ids[:self.stride] = -100

        return input_ids, target_ids


class PTBDataModule(LightningDataModule):
    def __init__(self, directory_dataset=os.getcwd(), batch_size=64, sequence_length=2048, stride=512, n_lines=None, tokenizer_name=None, seed=1):
        super().__init__()
        self.directory_dataset = directory_dataset
        self.batch_size = batch_size
        self.n_lines = n_lines
        self.stride = stride
        self.sequence_length = sequence_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, legacy=False)
        self.prepare_data()

    def prepare_data(self):
        # Load train, val, and test datasets
        train_split = "train" if self.n_lines is None else f"train[:{self.n_samples}]"
        validation_split = "validation" if self.n_lines is None else f"validation[:{self.n_samples}]"
        test_split = "test" if self.n_lines is None else f"test[:{self.n_samples}]"
        
        self.train_dataset = load_dataset('ptb_text_only', 'penn_treebank', split=train_split, trust_remote_code=True)
        self.val_dataset = load_dataset('ptb_text_only', 'penn_treebank', split=validation_split, trust_remote_code=True)
        self.test_dataset = load_dataset('ptb_text_only', 'penn_treebank', split=test_split, trust_remote_code=True)
        
        # Rename 'sentence' to 'text' for each dataset
        self.train_dataset = rename_feature(self.train_dataset, 'sentence', 'text')
        self.val_dataset = rename_feature(self.val_dataset, 'sentence', 'text')
        self.test_dataset = rename_feature(self.test_dataset, 'sentence', 'text')

    def train_dataloader(self, batch_size=None, sequence_length=None, stride=None):
        if batch_size is None:
            batch_size = self.batch_size
        if sequence_length is None:
            sequence_length = self.sequence_length
        else:
            sequence_length = min(self.sequence_length, sequence_length)
        if stride is None:
            stride = self.stride
        dataset = TextDataset(self.train_dataset, tokenizer=self.tokenizer, sequence_length=sequence_length, stride=stride)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        train_dataloader.ORIGINAL_DATASET = self.train_dataset
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
        dataset = TextDataset(self.val_dataset, tokenizer=self.tokenizer, sequence_length=sequence_length, stride=stride)
        val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_dataloader.ORIGINAL_DATASET = self.val_dataset
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
        dataset = TextDataset(self.test_dataset, tokenizer=self.tokenizer, sequence_length=sequence_length, stride=stride)
        test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        test_dataloader.ORIGINAL_DATASET = self.test_dataset
        return test_dataloader
