from abc import ABC
from dataclasses import asdict
from typing import List, Dict, Optional, Any, Union
import copy
import torch
import wandb
from src.loggers.wandb_utils import safe_wandb_log

from src.reliability_eval.calculators.base import ScoresCalculator
from src.reliability_eval.common.constants import TOKEN_SCORES_ID_FIELDS
from src.reliability_eval.common.score_types import QuestionAggregateTypes, SequenceAggregateTypes, SequenceScoreTypes
from src.reliability_eval.common.scores import QuestionAggregateScores, SequenceAggregatedScores, SequenceScores, TokenScores


class BaseAggregator(ScoresCalculator, ABC):
    """Base class for score aggregation with common functionality."""
    
    def __init__(
        self,
        model,
        tokenizer,
        inference_config: Dict = {}
    ):
        """Initialize aggregator with model and tokenizer."""
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            inference_config=inference_config,
        )

    def _aggregate_scores_with_aggregator(
        self,
        scores: Union[TokenScores, List[SequenceScores]],
        aggregator: callable,
        boolean_mask: Optional[torch.BoolTensor] = None,
        effective_lengths: Optional[torch.Tensor] = None
    ) -> Any:
        """Aggregate scores using specified aggregator."""
        aggregated_scores = copy.deepcopy(scores)
        self._process_scores(aggregated_scores, scores, aggregator, boolean_mask, effective_lengths)
        return aggregated_scores

    def _process_scores(
        self,
        aggregated_scores: Any,
        original_scores: Any,
        aggregator: callable,
        boolean_mask: Optional[torch.BoolTensor],
        effective_lengths: Optional[torch.Tensor] = None
    ) -> None:
        """Process all score fields for aggregation."""
        for field_name, field_value in asdict(original_scores).items():
            if field_value is None:
                continue
            if field_name == SequenceScoreTypes.IS_CORRECT.value:
                self._handle_correctness_scores(aggregated_scores, field_value, aggregator)
            elif field_name not in self._get_id_fields():
                self._process_score_field(aggregated_scores, field_name, field_value, aggregator, boolean_mask, effective_lengths)

    def _get_id_fields(self) -> List[str]:
        """Return list of fields that should not be aggregated."""
        return TOKEN_SCORES_ID_FIELDS

    def _handle_correctness_scores(
        self,
        aggregated_scores: Any,
        correctness_scores: torch.Tensor,
        aggregator: callable
    ) -> None:
        """Handle aggregation of correctness scores."""
        aggregated_values = self._apply_aggregator(correctness_scores.float(), aggregator)
        setattr(aggregated_scores, SequenceScoreTypes.IS_CORRECT.value, aggregated_values)

    def _process_score_field(
        self,
        aggregated_scores: Any,
        field_name: str,
        field_value: Any,
        aggregator: callable,
        boolean_mask: Optional[torch.BoolTensor],
        effective_lengths: Optional[torch.Tensor] = None
    ) -> None:
        """Process individual score field for aggregation."""
        if isinstance(field_value, dict):
            for score_name, score_value in field_value.items():
                if score_name not in self._get_id_fields() and score_value is not None:
                    aggregated_value = self._aggregate_score(score_value, aggregator, boolean_mask, effective_lengths)
                    self._update_aggregated_scores(aggregated_scores, field_name, score_name, aggregated_value)
        else:
            if field_value is None:
                setattr(aggregated_scores, field_name, None)
            elif field_value not in self._get_id_fields():
                aggregated_value = self._aggregate_score(field_value, aggregator, boolean_mask, effective_lengths)
                setattr(aggregated_scores, field_name, aggregated_value)
    
    def _aggregate_score(
        self,
        score_value: torch.Tensor,
        aggregator: callable,
        boolean_mask: Optional[torch.BoolTensor] = None,
        effective_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply appropriate aggregation to score values with effective length masking."""
        # Create a mask based on effective lengths if provided
        if effective_lengths is not None:
            max_len = score_value.shape[-1]
            
            # Create position indices [0, 1, 2, ..., max_len-1]
            positions = torch.arange(max_len, device=score_value.device)
            
            # Create a mask where True if position < effective_length
            effective_mask = positions.unsqueeze(0).unsqueeze(0) < effective_lengths.unsqueeze(-1)
            
            # Combine with existing boolean mask if provided
            if boolean_mask is not None:
                effective_mask = effective_mask & boolean_mask
            
            safe_wandb_log({
                "aggregation_effective_mask": effective_mask
            })
                
            return self._compute_masked_aggregation(score_value, effective_mask, aggregator)
        
        if boolean_mask is not None:
            return self._compute_masked_aggregation(score_value, boolean_mask, aggregator)
        
        return self._apply_aggregator(score_value, aggregator)
    
    def _compute_masked_aggregation(
        self,
        values: torch.Tensor,
        mask: torch.BoolTensor,
        aggregator: callable
    ) -> torch.Tensor:
        """Compute aggregation of values considering only masked elements."""
        masked_values = values * mask.float()
        if aggregator == torch.mean:
            valid_counts = mask.sum(dim=-1)
            return masked_values.sum(dim=-1) / valid_counts
        elif aggregator == torch.sum:
            return masked_values.sum(dim=-1)
        elif aggregator in [torch.max, torch.min]:
            # Set values outside mask to be extreme values so they don't affect min/max
            extreme_value = -float('inf') if aggregator == torch.max else float('inf')
            masked_values = torch.where(mask, values, torch.tensor(extreme_value, device=values.device))
            return aggregator(masked_values, dim=-1).values
        else:
            # For custom aggregators, apply to masked values
            return aggregator(masked_values, dim=-1)

    def _apply_aggregator(
        self,
        values: torch.Tensor,
        aggregator: callable
    ) -> torch.Tensor:
        """Apply aggregator function to values."""
        return aggregator(values, dim=-1).values if aggregator in [torch.max, torch.min] else aggregator(values, dim=-1)
        
    def _update_aggregated_scores(
        self,
        aggregated_scores: Any,
        field_name: str,
        score_name: str,
        aggregated_value: torch.Tensor
    ) -> None:
        """Update aggregated scores with new values."""
        target = getattr(aggregated_scores, field_name)
        setattr(target, score_name, aggregated_value)

class SequenceAggregator(BaseAggregator):
    """Aggregator for sequence-level scores."""

    def calculate_sequence_aggregate_scores(
        self,
        token_scores: TokenScores,
        sequence_aggregate_types: List[SequenceAggregateTypes]
    ) -> SequenceAggregatedScores:
        """Calculate aggregated metrics for sequences using effective lengths."""
        effective_lengths = token_scores.effective_lengths if hasattr(token_scores, 'effective_lengths') else None
        
        return SequenceAggregatedScores(
            mean_sequence_aggregate_scores=self._aggregate_if_type(
                token_scores, torch.mean, SequenceAggregateTypes.MEAN, sequence_aggregate_types, effective_lengths),
            total_sequence_aggregate_scores=self._aggregate_if_type(
                token_scores, torch.sum, SequenceAggregateTypes.SUM, sequence_aggregate_types, effective_lengths),
            max_sequence_aggregate_scores=self._aggregate_if_type(
                token_scores, torch.max, SequenceAggregateTypes.MAX, sequence_aggregate_types, effective_lengths),
            min_sequence_aggregate_scores=self._aggregate_if_type(
                token_scores, torch.min, SequenceAggregateTypes.MIN, sequence_aggregate_types, effective_lengths)
        )

    def _aggregate_if_type(
        self,
        scores: TokenScores,
        aggregator: callable,
        agg_type: SequenceAggregateTypes,
        allowed_types: List[SequenceAggregateTypes],
        effective_lengths: Optional[torch.Tensor] = None
    ) -> Optional[TokenScores]:
        """Aggregate scores if aggregation type is allowed."""
        if agg_type in allowed_types:
            return self._aggregate_scores_with_aggregator(
                scores, 
                aggregator,
                effective_lengths=effective_lengths
            )
        return None

class QuestionAggregator(BaseAggregator):
    """Aggregator for question-level scores."""
    
    def calculate_question_aggregate_scores(
        self,
        sequence_scores: List[SequenceScores],
        question_aggregate_types: List[QuestionAggregateTypes]
    ) -> QuestionAggregateScores:
        """Calculate aggregated metrics for questions."""
        return QuestionAggregateScores(
            mean_question_aggregate_scores=self._aggregate_if_type(
                sequence_scores, torch.mean, QuestionAggregateTypes.MEAN, question_aggregate_types),
            total_question_aggregate_scores=self._aggregate_if_type(
                sequence_scores, torch.sum, QuestionAggregateTypes.SUM, question_aggregate_types),
            max_question_aggregate_scores=self._aggregate_if_type(
                sequence_scores, torch.max, QuestionAggregateTypes.MAX, question_aggregate_types),
            min_question_aggregate_scores=self._aggregate_if_type(
                sequence_scores, torch.min, QuestionAggregateTypes.MIN, question_aggregate_types),
            accuracy_aggregate_scores=self._calculate_accuracy(
                sequence_scores) if QuestionAggregateTypes.ACCURACY in question_aggregate_types else None,
            mean_among_correct_aggregate_scores=self._calculate_mean_among_correct(
                sequence_scores) if QuestionAggregateTypes.MEAN_AMONG_CORRECT in question_aggregate_types else None
        )

    def _aggregate_if_type(
        self,
        scores: List[SequenceScores],
        aggregator: callable,
        agg_type: QuestionAggregateTypes,
        allowed_types: List[QuestionAggregateTypes]
    ) -> Optional[SequenceScores]:
        """Aggregate scores if aggregation type is allowed."""
        return self._aggregate_scores_with_aggregator(scores, aggregator) if agg_type in allowed_types else None

    def _calculate_accuracy(self, sequence_scores: List[SequenceScores]) -> torch.Tensor:
        """Calculate accuracy from sequence scores."""
        if not hasattr(sequence_scores, SequenceScoreTypes.IS_CORRECT.value):
            raise ValueError("Sequence scores must contain 'is_correct' field")
        return torch.mean(getattr(sequence_scores, SequenceScoreTypes.IS_CORRECT.value).float(), dim=-1)

    def _calculate_mean_among_correct(self, sequence_scores: List[SequenceScores]) -> SequenceScores:
        """Calculate mean scores among correct sequences."""
        if not hasattr(sequence_scores, SequenceScoreTypes.IS_CORRECT.value):
            raise ValueError("Sequence scores must contain 'is_correct' field")
        correct_mask = getattr(sequence_scores, SequenceScoreTypes.IS_CORRECT.value)
        return self._aggregate_scores_with_aggregator(sequence_scores, torch.mean, correct_mask)