from typing import List, Optional
import pandas as pd
from src.dataset_processing.datasets.mmlu.models import MMLUEntry, ProcessedMMLUEntry
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
from src.dataset_processing.perturbations.enums import PerturbationType
from src.dataset_processing.perturbations.utils.registry import create_perturbation


class MMLUDataProcessor:
    """Handles data processing for MMLU datasets."""
    
    @staticmethod
    def get_task_prefix() -> str:
        """Get the task description prefix."""
        return "The following are multiple choice questions (with answers).\n\n"

    @staticmethod
    def parse_raw_entry(raw_entry: pd.Series, subject: str, idx: int) -> MMLUEntry:
        """Convert raw DataFrame row to MMLUEntry."""
        return MMLUEntry(
            question=raw_entry["question"],
            choices=[raw_entry[choice] for choice in ("A", "B", "C", "D")],
            answer_key=raw_entry.get("answer"),
            subject=subject,
            question_id=f"{subject}_{idx}"
        )

    @staticmethod
    def apply_perturbations(text: str, perturbation_config: Optional[PerturbationConfig]) -> str:
        """Apply perturbation modifications to text."""
        if perturbation_config is None or perturbation_config.type == PerturbationType.NONE:
            return text
        perturber = create_perturbation(perturbation_config=perturbation_config)
        return perturber.perturb(text)

    @staticmethod
    def format_question(entry: MMLUEntry, include_prefix: bool = True) -> str:
        """Format question with choices according to template."""
        prefix = MMLUDataProcessor.get_task_prefix() if include_prefix else ""
        formatted_question = f"""Question: {entry.question.strip()}
A. {entry.choices[0]}
B. {entry.choices[1]}
C. {entry.choices[2]}
D. {entry.choices[3]}
Answer:"""
        return prefix + formatted_question

    @staticmethod
    def format_few_shot_example(entry: MMLUEntry) -> str:
        """Format a few-shot example including the correct answer."""
        formatted_question = f"""Question: {entry.question.strip()}
A. {entry.choices[0]}
B. {entry.choices[1]}
C. {entry.choices[2]}
D. {entry.choices[3]}
Answer: {entry.answer_key}

"""
        return formatted_question

    @staticmethod
    def create_few_shot_prompt(examples: List[MMLUEntry], num_examples: int = 3) -> str:
        """Create a few-shot prompt from examples."""
        if not examples:
            return ""
        
        prompt = MMLUDataProcessor.get_task_prefix()
        for example in examples[:num_examples]:
            prompt += MMLUDataProcessor.format_few_shot_example(example)
        return prompt

    @staticmethod
    def create_processed_entry(
        entry: MMLUEntry,
        perturbation_config: Optional[PerturbationConfig] = None,
        few_shot_examples: Optional[List[MMLUEntry]] = None
    ) -> ProcessedMMLUEntry:
        """Create processed entry with optional few-shot examples and perturbations."""
        formatted_question = MMLUDataProcessor.format_question(
            entry,
            include_prefix=not bool(few_shot_examples)
        )
        
        if few_shot_examples:
            few_shot_prompt = MMLUDataProcessor.create_few_shot_prompt(few_shot_examples)
            formatted_question = few_shot_prompt + formatted_question
            
        if perturbation_config:
            formatted_question = MMLUDataProcessor.apply_perturbations(
                formatted_question,
                perturbation_config
            )

        metadata = {
            'id': entry.question_id,
            'original_question': entry.question,
            'subject': entry.subject,
            'num_few_shot_examples': len(few_shot_examples) if few_shot_examples else 0
        }

        return ProcessedMMLUEntry(
            question=formatted_question,
            answer=entry.answer_key,
            metadata=metadata
        )
