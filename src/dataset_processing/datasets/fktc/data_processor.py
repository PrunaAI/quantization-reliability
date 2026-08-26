from typing import Dict, List, Optional
from src.dataset_processing.datasets.fktc.models import FKTCEntry, ProcessedFKTCEntry
from src.dataset_processing.perturbations.config.perturbation_config import PerturbationConfig
from src.dataset_processing.perturbations.enums import PerturbationType
from src.dataset_processing.perturbations.utils.registry import create_perturbation


class FKTCDataProcessor:
    """Handles data processing for FKTC datasets"""
    
    @staticmethod
    def parse_raw_entry(raw_entry: Dict) -> FKTCEntry:
        """Convert raw JSON entry to FKTCEntry"""
        return FKTCEntry(
            subject=raw_entry['subject'],
            object=raw_entry['object'],
            taxonomy=raw_entry.get('taxonomy', []),
            relations=raw_entry.get('relations', [])
        )
        
    @staticmethod
    def get_few_shot_examples(df, num_shots: int = 0) -> List[FKTCEntry]:
        """Get few-shot examples from the first entries."""
        if num_shots == 0:
            return []
        examples = []
        for _, row in df.head(num_shots).iterrows():
            entry = FKTCDataProcessor.parse_raw_entry({
                'subject': row['subject'],
                'object': row['answer'],
                'taxonomy': row['taxonomy'].split(',') if row['taxonomy'] else []
            })
            examples.append(entry)
        return examples

    @staticmethod
    def format_few_shot_example(entry: FKTCEntry, relation: str) -> str:
        """Format a few-shot example including the correct answer."""
        question = relation.replace("[X]", entry.subject)
        return f"Question: {question}\nAnswer: {entry.object}\n\n"
        
    @staticmethod
    def apply_perturbations(text: str, perturbation_config: Optional[PerturbationConfig]) -> str:
        """Apply perturbation modifications to text"""
        if perturbation_config is None or perturbation_config.type == PerturbationType.NONE:
            return text
        
        perturber = create_perturbation(perturbation_config=perturbation_config)
        
        return perturber.perturb(text)
    
    @staticmethod
    def create_processed_entry(
        entry: FKTCEntry,
        relation: str,
        perturbation_config: Optional[PerturbationConfig] = None,
        few_shot_examples: Optional[List[FKTCEntry]] = None
    ) -> ProcessedFKTCEntry:
        """Create processed entry from raw entry and relation"""
        # Add the actual question
        question = relation.replace("[X]", entry.subject)
        
        if perturbation_config:
            perturbation_config.taxonomies = entry.taxonomy + [entry.object]
            question = FKTCDataProcessor.apply_perturbations(
                question, perturbation_config
            )
        
        # Format few-shot examples if provided
        # question_prefix = 'This is a bot that correctly answers questions. \n\n'
        question_prefix = ''
        if few_shot_examples:
            for example in few_shot_examples:
                question_prefix += FKTCDataProcessor.format_few_shot_example(example, relation)
        
        question_postfix = '\nAnswer:'
        formatted_question = question_prefix + "Question: " + question + question_postfix

        return ProcessedFKTCEntry(
            question=formatted_question,
            answer=entry.object,
            metadata={
                'subject': entry.subject,
                'relation': relation,
                'taxonomy': entry.taxonomy,
                'num_few_shot_examples': len(few_shot_examples) if few_shot_examples else 0
            }
        )
