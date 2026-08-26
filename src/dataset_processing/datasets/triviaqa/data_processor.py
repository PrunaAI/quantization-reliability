from typing import Dict, List, Optional
import datasets
from src.dataset_processing.datasets.triviaqa.models import ProcessedTriviaQAEntry, TriviaQAEntry
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
from src.dataset_processing.perturbations.enums import PerturbationType
from src.dataset_processing.perturbations.utils.registry import create_perturbation


class TriviaQADataProcessor:
    """Handles data processing for TriviaQA datasets."""
    
    @staticmethod
    def parse_raw_entry(raw_entry: Dict) -> TriviaQAEntry:
        """Convert raw HuggingFace entry to TriviaQAEntry."""
        answer_list = []
        answer_list.append(raw_entry['answer']['normalized_value'])
        answer_list.extend(raw_entry['answer']['normalized_aliases'])
        
        return TriviaQAEntry(
            question=raw_entry['question'],
            answer_list=answer_list,
            question_id=raw_entry['question_id']
        )
        
    @staticmethod
    def get_few_shot_examples(num_shots: int = 0) -> List[TriviaQAEntry]:
        """Get few-shot examples from the training data."""
        if num_shots == 0:
            return []
        
        # Load first 10 examples from training data
        train_data = datasets.load_dataset("trivia_qa", "rc.nocontext", split="train")
        examples = []
        for example in train_data.select(range(0, num_shots)):
            entry = TriviaQADataProcessor.parse_raw_entry(example)
            examples.append(entry)
        return examples
    
    @staticmethod
    def format_few_shot_example(entry: TriviaQAEntry) -> str:
        """Format a few-shot example including the correct answer."""
        return f"Question: {entry.question}\nAnswer: {entry.answer_list[0]}\n\n"
    
    @staticmethod
    def apply_perturbations(text: str, perturbation_config: Optional[PerturbationConfig], answer_list: List[str]) -> str:
        """Apply perturbation modifications to text."""
        if perturbation_config is None or perturbation_config.type == PerturbationType.NONE:
            return text
            
        perturber = create_perturbation(perturbation_config=perturbation_config)
        return perturber.perturb(text)
    
    @staticmethod
    def create_processed_entry(
        entry: TriviaQAEntry,
        perturbation_config: Optional[PerturbationConfig] = None,
        few_shot_examples: Optional[List[TriviaQAEntry]] = None
    ) -> ProcessedTriviaQAEntry:
        """Create processed entry from raw entry."""       
        # Add the actual question
        formatted_question = f"Question: {entry.question}"
        
        if perturbation_config:
            formatted_question = TriviaQADataProcessor.apply_perturbations(
                formatted_question,
                perturbation_config,
                entry.answer_list
            )
            
        # Format few-shot examples if provided
        # question_prefix = 'Answer the following question as briefly as possible. \n'
        question_prefix = ''
        if few_shot_examples:
            for example in few_shot_examples:
                question_prefix += TriviaQADataProcessor.format_few_shot_example(example)
        
        question_postfix = '\nAnswer:'
        formatted_question = question_prefix + formatted_question + question_postfix
        
        return ProcessedTriviaQAEntry(
            question=formatted_question,
            answer='|'.join(entry.answer_list),
            metadata={
                'id': entry.question_id,
                'original_question': entry.question,
                'answer_list': entry.answer_list,
                'num_few_shot_examples': len(few_shot_examples) if few_shot_examples else 0
            }
        )
