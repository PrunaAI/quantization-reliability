from typing import List, Set, Optional, Tuple
import random
from deep_translator import GoogleTranslator # type: ignore
from dataclasses import dataclass

from src.dataset_processing.perturbations.base.text_perturbation import TextPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig

@dataclass
class TranslationConfig:
    """Configuration for translation operations."""
    max_failed_attempts: int = 5
    max_iterations: int = 100
    languages: Set[str] = frozenset(['es', 'fr', 'de', 'it', 'ru', 'zh-cn', 'ja'])
    min_languages: int = 2

class TranslationManager:
    """Handler for translation operations."""
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.chosen_languages: List[str] = []
    
    def get_translation_language(self) -> str:
        """Get target language for translation."""
        if len(self.chosen_languages) < self.config.min_languages:
            lang = random.choice(list(self.config.languages - set(self.chosen_languages)))
            self.chosen_languages.append(lang)
        else:
            lang = random.choice(self.chosen_languages)
        return lang
    
    def translate_text(self, text: str, target_lang: str) -> Optional[str]:
        """Translate text to target language."""
        try:
            translator = GoogleTranslator(source='auto', target=target_lang)
            return translator.translate(text)
        except Exception as e:
            return None

class PhraseTranslation(TextPerturbation):
    """Implementation of random phrase translation."""
    
    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.translation_manager = TranslationManager(TranslationConfig())
        self.translated_indices = set()
    
    def _get_phrase_indices(self, words: List[str]) -> Optional[Tuple[int, int]]:
        """Get valid indices for phrase translation."""
        if len(words) < 2:
            return None
            
        # Find consecutive indices
        valid_starts = range(len(words) - 1)
        if not valid_starts:
            return None
            
        start_idx = random.choice(list(valid_starts))
        return start_idx, start_idx + 1
    
    def _translate_unit(self, words: List[str], is_phrase: bool) -> bool:
        try:
            if is_phrase:
                indices = self._get_phrase_indices(words)
                if indices:
                    start_idx, end_idx = indices
                    # Check if any word in phrase was already translated
                    if any(i in self.translated_indices for i in range(start_idx, end_idx + 1)):
                        return False
                    phrase = ' '.join(words[start_idx:end_idx + 1])
                    translated = self.translation_manager.translate_text(
                        phrase,
                        self.translation_manager.get_translation_language()
                    )
                    
                    if translated:
                        translated_words = translated.split()
                        if translated_words:
                            words[start_idx:end_idx + 1] = translated_words
                            # Add all indices in phrase to translated set
                            self.translated_indices.update(range(start_idx, end_idx + 1))
                            return True
            else:
                available_indices = [i for i in range(len(words)) if i not in self.translated_indices]
                if not available_indices:
                    return False
                idx = random.choice(available_indices)
                translated = self.translation_manager.translate_text(
                    words[idx],
                    self.translation_manager.get_translation_language()
                )
                
                if translated:
                    words[idx] = translated
                    self.translated_indices.add(idx)
                    return True
                    
        except Exception as e:
            return False
                
        return False
    
    def perturb(self, text: str) -> str:
        """Apply random phrase translation to text."""
        self.translated_indices.clear()
        
        # Split into question and answer parts
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text
            
        # Process only question part
        words = question_part.split()
        if not words:
            return text
            
        translations_applied = 0
        failed_attempts = 0
        iterations = 0
        
        max_iterations = getattr(TranslationConfig, 'max_iterations', 100)
        max_failed_attempts = getattr(TranslationConfig, 'max_failed_attempts', 5)
        
        while (
            translations_applied < self.config.intensity and 
            iterations < max_iterations and
            failed_attempts < max_failed_attempts
        ):
            # Only attempt phrase translation if we have enough words
            is_phrase = random.choice([True, False]) if len(words) > 1 else False
            
            try:
                success = self._translate_unit(words, is_phrase)
                if success:
                    translations_applied += 1
                else:
                    failed_attempts += 1
            except Exception as e:
                failed_attempts += 1
                
            iterations += 1
        
        # Combine processed question with unchanged answer part
        return " ".join(words) + answer_part
