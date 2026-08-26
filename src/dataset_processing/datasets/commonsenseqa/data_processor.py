from typing import Dict, List, Optional, Any

from src.dataset_processing.common.constants import SEVEN_SHOT_EXAMPLES
from src.dataset_processing.datasets.commonsenseqa.models import CommonSenseQAEntry, ProcessedCommonSenseQAEntry
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
from src.dataset_processing.perturbations.enums import PerturbationType
from src.dataset_processing.perturbations.utils.registry import create_perturbation


class CommonSenseQADataProcessor:
    """Handles data processing for CommonSenseQA datasets with improved formatting and few-shot support."""
    
    @staticmethod
    def get_task_prefix() -> str:
        """Get the task description prefix."""
        # return "This is a bot that correctly answers questions. \n\n"
        return ""
    
    @staticmethod
    def parse_raw_entry(raw_entry: Dict) -> CommonSenseQAEntry:
        """Convert raw HuggingFace entry to CommonSenseQAEntry."""
        return CommonSenseQAEntry(
            question=raw_entry['question'],
            choices=raw_entry['choices'],
            answer_key=raw_entry['answerKey'],
            question_id=raw_entry['id']
        )
    
    @staticmethod
    def apply_perturbations(text: str, perturbation_config: Optional[PerturbationConfig]) -> str:
        """Apply perturbation modifications to text."""
        if perturbation_config is None or perturbation_config.type == PerturbationType.NONE:
            return text
        perturber = create_perturbation(perturbation_config=perturbation_config)
        return perturber.perturb(text)
    
    @staticmethod
    def format_question(entry: CommonSenseQAEntry) -> str:
        """
        Format question with choices according to template.
        """
        return f"Question: {entry.question.strip()}\nA. {entry.choices['text'][0]}\nB. {entry.choices['text'][1]}\nC. {entry.choices['text'][2]}\nD. {entry.choices['text'][3]}\nE. {entry.choices['text'][4]}\n"
    
    @staticmethod
    def format_few_shot_example(entry: CommonSenseQAEntry, include_explanation: bool = False) -> str:
        """Format a few-shot example including the correct answer."""
        # Start with question
        formatted_question = f"Question: {entry.question.strip()}\n"
        
        # Add choices dynamically based on how many are available
        for i, choice in enumerate(entry.choices['text']):
            letter = chr(65 + i)  # A, B, C, etc.
            formatted_question += f"{letter}. {choice}\n"
        
        if include_explanation and hasattr(entry, 'explanation'):
            formatted_question += f"Answer: {entry.explanation}\nSo the answer is {entry.answer_key}.\n\n"
        else:
            formatted_question += f"Answer: {entry.answer_key}\n\n"
            
        return formatted_question
    
    @staticmethod
    def get_few_shot_examples(num_shots: int = 0) -> List[CommonSenseQAEntry]:
        """Get few-shot examples based on number of shots requested."""
        if num_shots == 0:
            return []
        
        examples = []
        for example in SEVEN_SHOT_EXAMPLES[:num_shots]:
            entry = CommonSenseQAEntry(
                question=example["question"],
                choices=example["choices"],
                answer_key=example["answer_key"],
                question_id="few_shot_example"
            )
            # Add explanation as an attribute
            setattr(entry, 'explanation', example["explanation"])
            examples.append(entry)
        return examples

    @staticmethod
    def create_few_shot_prompt(examples: List[CommonSenseQAEntry], num_examples: int = 3) -> str:
        """
        Create a few-shot prompt from examples.
        """
        if not examples:
            return ""
        
        prompt = CommonSenseQADataProcessor.get_task_prefix()
        for example in examples[:num_examples]:
            prompt += CommonSenseQADataProcessor.format_few_shot_example(example)
        return prompt
    
    @staticmethod
    def create_processed_entry(
        entry: CommonSenseQAEntry,
        perturbation_config: Optional[PerturbationConfig] = None,
        few_shot_examples: Optional[List[CommonSenseQAEntry]] = None,
        num_shots: int = 0
    ) -> ProcessedCommonSenseQAEntry:
        """
        Create processed entry with optional few-shot examples and perturbations.
        """
        if few_shot_examples is None and num_shots > 0:
            few_shot_examples = CommonSenseQADataProcessor.get_few_shot_examples(num_shots)
        
        formatted_question = CommonSenseQADataProcessor.format_question(entry)
        correct_answer = entry.answer_key
        
        if perturbation_config:
            formatted_question = CommonSenseQADataProcessor.apply_perturbations(
                formatted_question,
                perturbation_config
            )
            
        if few_shot_examples:
            few_shot_prompt = ""
            for example in few_shot_examples:
                few_shot_prompt += CommonSenseQADataProcessor.format_few_shot_example(
                    example,
                    include_explanation=(num_shots > 0)  # Include explanations for standard few-shot examples
                )
            formatted_question = few_shot_prompt + formatted_question
        
        question_prefix = CommonSenseQADataProcessor.get_task_prefix()
        formatted_question = question_prefix + formatted_question
        
        metadata = {
            'id': entry.question_id,
            'original_question': entry.question,
            'num_few_shot_examples': len(few_shot_examples) if few_shot_examples else 0
        }
        
        return ProcessedCommonSenseQAEntry(
            question=formatted_question,
            answer=correct_answer,
            metadata=metadata
        )
