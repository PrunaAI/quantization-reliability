from typing import List, Optional, Set
import random
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import nltk

# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

from src.dataset_processing.perturbations.base.word_perturbation import WordPerturbation
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
from src.dataset_processing.perturbations.utils.word_processor import WordProcessor

class WordNetSynonymFetcher:
    """Handler for WordNet-based synonym operations."""
    
    # POS mapping from NLTK to WordNet
    POS_MAP = {
        'NN': wordnet.NOUN,
        'NNS': wordnet.NOUN,
        'VB': wordnet.VERB,
        'VBD': wordnet.VERB,
        'VBG': wordnet.VERB,
        'VBN': wordnet.VERB,
        'VBP': wordnet.VERB,
        'VBZ': wordnet.VERB,
        'JJ': wordnet.ADJ,
        'JJR': wordnet.ADJ,
        'JJS': wordnet.ADJ,
        'RB': wordnet.ADV,
        'RBR': wordnet.ADV,
        'RBS': wordnet.ADV
    }
    
    # Words to avoid replacing
    SKIP_WORDS: Set[str] = {
        'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'of',
        'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
        'off', 'over', 'under', 'answer', 'question'
    }

    def get_wordnet_pos(self, nltk_tag: str) -> Optional[str]:
        """Convert NLTK POS tag to WordNet POS tag."""
        return self.POS_MAP.get(nltk_tag)

    def get_synonyms(self, word: str, pos: Optional[str] = None) -> List[str]:
        """Get synonyms for a word with optional POS."""
        if word.lower() in self.SKIP_WORDS:
            return []

        synonyms = set()
        for synset in wordnet.synsets(word, pos=pos):
            for lemma in synset.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.add(lemma.name())
        return list(synonyms)

    def get_best_synonym(self, word: str, pos: Optional[str] = None) -> Optional[str]:
        """Get the most appropriate synonym for a word."""
        synonyms = self.get_synonyms(word, pos)
        if not synonyms:
            return None
            
        # Filter out multi-word synonyms, those too similar to original,
        # and those containing newlines or special characters
        valid_synonyms = [
            syn for syn in synonyms 
            if (len(syn.split()) == 1 and 
                not (word.lower() in syn.lower() or syn.lower() in word.lower()) and
                '\n' not in syn and
                syn.isalnum())  # Only allow alphanumeric synonyms
        ]
        
        return random.choice(valid_synonyms) if valid_synonyms else None

class SynonymReplacement(WordPerturbation):
    """Implementation of WordNet-based synonym replacement."""
    
    def __init__(self, config: PerturbationConfig):
        super().__init__(config)
        self.word_processor = WordProcessor()
        self.synonym_fetcher = WordNetSynonymFetcher()

    def get_candidates(self, word: str) -> List[str]:
        """Get synonym candidates for a word."""
        # Get POS tag for the word
        pos_tag_obj = pos_tag([word])[0][1]
        wordnet_pos = self.synonym_fetcher.get_wordnet_pos(pos_tag_obj)
        return self.synonym_fetcher.get_synonyms(word, wordnet_pos)

    def select_replacement(self, word: str, candidates: List[str]) -> Optional[str]:
        """Select appropriate synonym replacement."""
        # Filter out multi-word synonyms and those too similar to original
        valid_synonyms = [
            syn for syn in candidates 
            if (len(syn.split()) == 1 and 
                not (word.lower() in syn.lower() or syn.lower() in word.lower()) and
                '\n' not in syn and
                syn.isalnum())
        ]
        return random.choice(valid_synonyms) if valid_synonyms else None

    def _is_answer_word(self, word: str, idx: int, words: List[str]) -> bool:
        """Check if a word is part of an answer choice."""
        # Check if word is immediately after an answer prefix (A., B., etc.)
        if idx > 0:
            prev_word = words[idx - 1]
            if (len(prev_word) == 2 and 
                prev_word[0] in 'ABCDE' and 
                prev_word[1] == '.'):
                return True
        return False

    def _is_safe_for_replacement(self, word: str, idx: int, words: List[str]) -> bool:
        """Check if it's safe to replace this word."""
        # Skip if word contains newline
        if '\n' in word:
            return False
            
        # Skip answer choices
        if self._is_answer_word(word, idx, words):
            return False
            
        # Skip special words and patterns
        if (word.lower() in {'question:', 'answer:', 'a.', 'b.', 'c.', 'd.', 'e.'} or
            word.startswith(('A.', 'B.', 'C.', 'D.', 'E.'))):
            return False
            
        return True

    def _get_replaceable_words(self, words: List[str]) -> List[tuple]:
        """Identify words that can be replaced with synonyms."""
        # Get POS tags for context-aware replacement
        tagged_words = pos_tag(words)
        
        replaceable_words = []
        perturbable_indices = self.get_perturbable_indices(len(words))
        
        for idx, (word, pos_tag_obj) in enumerate(tagged_words):
            if (idx not in perturbable_indices or 
                not self._is_safe_for_replacement(word, idx, words)):
                continue
                
            clean_word, start_punct, end_punct = self.word_processor.extract_punctuation(word)
            
            # Skip if the word has newlines in punctuation
            if '\n' in (start_punct or '') or '\n' in (end_punct or ''):
                continue
                
            wordnet_pos = self.synonym_fetcher.get_wordnet_pos(pos_tag_obj)
            
            if wordnet_pos:  # Only consider words with valid POS tags
                synonym = self.synonym_fetcher.get_best_synonym(clean_word, wordnet_pos)
                if synonym:
                    replaceable_words.append((
                        idx, word, clean_word, start_punct, end_punct, synonym
                    ))
                    
        return replaceable_words

    def perturb(self, text: str) -> str:
        """Apply synonym replacement to text."""
        # Split into question and answer parts
        question_part, answer_part = self.split_question_answer(text)
        
        if not question_part:
            return text
            
        # Process only question part
        words = question_part.split()
        if not words:
            return text
            
        # Get POS tags for context-aware replacement
        tagged_words = nltk.pos_tag(words)  # Use fully qualified nltk.pos_tag
        replaceable_words = []
        
        for idx, (word, tag) in enumerate(tagged_words):
            clean_word, start_punct, end_punct = self.word_processor.extract_punctuation(word)
            wordnet_pos = self.synonym_fetcher.get_wordnet_pos(tag)
            
            if wordnet_pos:  # Only consider words with valid POS tags
                synonym = self.synonym_fetcher.get_best_synonym(clean_word, wordnet_pos)
                if synonym:
                    replaceable_words.append((
                        idx, word, clean_word, start_punct, end_punct, synonym
                    ))
                    
        # Randomly select words to replace based on intensity
        num_replacements = min(self.config.intensity, len(replaceable_words))
        if num_replacements == 0:
            return text
            
        # Perform replacements
        result_words = words.copy()
        selected_indices = random.sample(range(len(replaceable_words)), num_replacements)
        
        for idx in selected_indices:
            word_idx, _, _, start_punct, end_punct, synonym = replaceable_words[idx]
            result_words[word_idx] = self.word_processor.restore_punctuation(
                synonym,
                start_punct,
                end_punct
            )
            
        # Combine processed question with unchanged answer part
        return " ".join(result_words) + answer_part
