from typing import Dict, Optional
from src.dataset_processing.datasets.coqa.models import CoQAEntry, ProcessedCoQAEntry
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
from src.dataset_processing.perturbations.enums import PerturbationType
from src.dataset_processing.perturbations.utils.registry import create_perturbation


class CoQADataProcessor:
    """Handles data processing for CoQA datasets."""
    
    @staticmethod
    def parse_raw_entry(
        story: str,
        question: Dict,
        answer: Dict,
        story_id: str
    ) -> CoQAEntry:
        """Convert raw JSON entry to CoQAEntry."""
        return CoQAEntry(
            story=story,
            question=question['input_text'],
            answer=answer['input_text'],
            question_id=question.get('turn_id', 0),
            story_id=story_id
        )
    
    @staticmethod
    def apply_perturbations(text: str, perturbation_config: Optional[PerturbationConfig]) -> str:
        """Apply perturbation modifications to text."""
        if perturbation_config is None or perturbation_config.type == PerturbationType.NONE:
            return text
            
        perturber = create_perturbation(perturbation_config=perturbation_config)
        return perturber.perturb(text)
    
    @staticmethod
    def create_processed_entry(
        entry: CoQAEntry,
        qa_history: str = "",
        perturbation_config: Optional[PerturbationConfig] = None
    ) -> ProcessedCoQAEntry:
        """Create processed entry from raw entry."""
        question = entry.question
        if perturbation_config:
            question = CoQADataProcessor.apply_perturbations(question, perturbation_config)
            
        if qa_history:
            question = f"{entry.story}{qa_history}\n\nQ: {question}"
        else:
            question = f"{entry.story}\n\nQ: {question}"
            
        question_prefix = ""
        question_postfix = f"\nA:"
        question = question_prefix + question + question_postfix
            
        return ProcessedCoQAEntry(
            question=question,
            answer=entry.answer,
            metadata={
                'story_id': entry.story_id,
                'question_id': entry.question_id,
                'original_question': entry.question
            }
        )
