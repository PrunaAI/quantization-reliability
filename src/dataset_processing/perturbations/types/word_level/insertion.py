from dataclasses import dataclass
from typing import List, Optional, Tuple
import random
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
from src.dataset_processing.perturbations.utils.word_processor import WordProcessor

@dataclass
class MLModelConfig:
    """Configuration for the language model."""
    model_name: str = "roberta-base"
    device: str = "cpu"

class LanguageModel:
    """Handler for masked language model operations."""
    def __init__(self, config: MLModelConfig):
        self.tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
        self.model = RobertaForMaskedLM.from_pretrained(config.model_name)
        self.model.to(config.device)
        self.model.eval()

    def predict_masked_token(self, sentence: str) -> str:
        """Predict word for masked position in sentence."""
        inputs = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            mask_token_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
            top_k = 10
            topk_tokens = outputs.logits[0, mask_token_index].topk(top_k)
            
            for token_id in topk_tokens.indices[0]:
                predicted_token = self.tokenizer.decode(token_id).strip()
                if '\n' not in predicted_token and predicted_token:
                    return predicted_token
            return ""

    @property
    def mask_token(self) -> str:
        """Get the model's mask token."""
        return self.tokenizer.mask_token

class WordInsertion(TextPerturbation):
    """Implementation of context-aware word insertion."""
    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.word_processor = WordProcessor()
        self.language_model = LanguageModel(MLModelConfig())

    def _create_masked_sentence(self, words: List[str], insert_position: int) -> str:
        """Create a sentence with mask token at insertion position."""
        if insert_position == len(words):
            masked_words = words + [self.language_model.mask_token]
        else:
            masked_words = words[:insert_position] + [self.language_model.mask_token] + words[insert_position:]
        return " ".join(masked_words)

    def _get_insertion_positions(self, text_length: int, num_insertions: int) -> List[tuple[int, int]]:
        """Generate random positions and counts for word insertion."""
        if text_length < 0 or num_insertions <= 0:
            return []
            
        max_position = text_length + 1  # +1 to allow insertion at end
        distribution = [0] * max_position
        
        # Randomly distribute all insertions
        for _ in range(num_insertions):
            position = random.randrange(max_position)
            distribution[position] += 1
        
        # Return only positions that got at least one insertion
        return [(pos, count) for pos, count in enumerate(distribution) if count > 0]

    def perturb(self, text: str) -> str:
        """Apply word insertion perturbation to text."""
        # Split into question and answer parts
        question_part, answer_part = self.split_question_answer(text)
        
        # Process only question part
        if not question_part:
            return text
            
        # Extract punctuation from words
        words = question_part.split()
        processed_words = [self.word_processor.extract_punctuation(word) for word in words]
        clean_words = [word[0] for word in processed_words]
        
        # Perform insertions
        position_counts = self._get_insertion_positions(len(clean_words), self.config.intensity)
        offset = 0
        for position, count in sorted(position_counts):  # Sort by position
            for _ in range(count):
                masked_sentence = self._create_masked_sentence(clean_words, position + offset)
                predicted_word = self.language_model.predict_masked_token(masked_sentence)
                if predicted_word:
                    clean_words.insert(position + offset, predicted_word)
                    processed_words.insert(position + offset, (predicted_word, None, None))
                    offset += 1
        
        # Restore punctuation
        result_words = [
            self.word_processor.restore_punctuation(word, start_punct, end_punct)
            for word, start_punct, end_punct in processed_words
        ]
        
        # Join words and combine with answer part
        return " ".join(result_words) + answer_part