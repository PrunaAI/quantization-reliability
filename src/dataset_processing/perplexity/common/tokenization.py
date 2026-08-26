from transformers import PreTrainedTokenizer
import torch
from dataclasses import dataclass
from typing import List

from src.dataset_processing.perplexity.common.models.dataset_entry import PerplexityDatasetEntry

@dataclass
class TokenizedOutput:
    """Container for tokenized text data."""
    input_ids: torch.Tensor
    target_ids: torch.Tensor

class PerplexityTokenizer:
    """Handles tokenization of WikiText data."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        
    def tokenize_texts(self, texts: List[str]) -> TokenizedOutput:
        """Tokenizes concatenated texts and prepares input/target pairs."""
        tokenized = self.tokenizer("\n\n".join(texts), return_tensors="pt")
        return TokenizedOutput(
            input_ids=tokenized.input_ids[0, :-1],
            target_ids=tokenized.input_ids[0, 1:]
        )
    
    def tokenize_qa_pairs(self, questions: List[str], answers: List[str], max_length: int) -> List[PerplexityDatasetEntry]:
        """Tokenizes question-answer pairs for Polyglot dataset."""
        entries = []
        for q, a in zip(questions, answers):
            # Tokenize question and answer separately
            q_tokens = self.tokenizer(q, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
            a_tokens = self.tokenizer(a, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0, 1:]  # Skip start token
            
            # Combine and create input/target pair
            input_ids = torch.cat((q_tokens, a_tokens))
            target_ids = torch.cat((q_tokens, a_tokens))
            target_ids[:len(q_tokens)] = -100  # Mask question tokens in target
            
            entries.append(PerplexityDatasetEntry(
                input_ids=input_ids[:-1],
                target_ids=target_ids[1:],
                metadata={"type": "qa_pair"}
            ))
        return entries