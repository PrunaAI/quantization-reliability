from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class TokenScores:
    """Container for sequence-level scores"""
    token_info_tuples: Optional[List[List[List[Tuple[int, str, float]]]]] = None  # [batch_size][num_outputs][tokens]
    full_decoded_text: Optional[List[List[str]]] = None  # [batch_size][num_outputs]
    relevant_decoded_text: Optional[List[List[str]]] = None   # [batch_size][num_outputs]
    cross_entropy_confidence_scores: Optional[torch.FloatTensor] = None   # [batch_size][num_outputs][tokens]
    negative_log_likelihood_scores: Optional[torch.FloatTensor] = None   # [batch_size][num_outputs][tokens]
    entropy_scores: Optional[torch.FloatTensor] = None   # [batch_size][num_outputs][tokens]
    top_k_concentration_scores: Optional[torch.FloatTensor] = None   # [batch_size][num_outputs][tokens]
    effective_lengths: Optional[torch.Tensor] = None   # [batch_size][num_outputs]
    padded_sequences: Optional[torch.Tensor] = None    # [batch_size][num_outputs][tokens]
    semantic_entropy_scores: Optional[torch.FloatTensor] = None # [batch_size]
    log_likelihood_raw_scores: Optional[torch.FloatTensor] = None # [batch_size][num_outputs][tokens]
    
@dataclass
class SequenceAggregatedScores:
    """Container for question-level scores"""
    mean_sequence_aggregate_scores: Optional[TokenScores]  # Shape: (batch_size, num_repeats * num_beams, )]
    total_sequence_aggregate_scores: Optional[TokenScores]  # Shape: (batch_size, num_repeats * num_beams, )]
    max_sequence_aggregate_scores: Optional[TokenScores]  # Shape: (batch_size, num_repeats * num_beams, )]
    min_sequence_aggregate_scores: Optional[TokenScores]  # Shape: (batch_size, num_repeats * num_beams, )]
    
@dataclass
class SequenceScores:
    """Container for question-level scores"""
    # Aggregated token scores    
    mean_sequence_aggregate_scores: Optional[TokenScores]  # Shape: (batch_size, num_repeats * num_beams, )]
    total_sequence_aggregate_scores: Optional[TokenScores]  # Shape: (batch_size, num_repeats * num_beams, )]
    max_sequence_aggregate_scores: Optional[TokenScores]  # Shape: (batch_size, num_repeats * num_beams, )]
    min_sequence_aggregate_scores: Optional[TokenScores]  # Shape: (batch_size, num_repeats * num_beams, )]
    
    # Correctness boolean values
    is_correct: Optional[torch.BoolTensor]  # Shape: (batch_size, num_repeats * num_beams, )]
    
    # Other scores
    # correct_entropy: Optional[torch.FloatTensor]  # Shape: (batch_size, num_repeats * num_beams, )]
    # answer_prob: Optional[torch.FloatTensor]  # Shape: (batch_size, num_repeats * num_beams, )]
    
@dataclass
class QuestionAggregateScores:
    """Container for question-level scores"""
    # Sequence aggregated scores
    mean_question_aggregate_scores: Optional[SequenceScores]  # Shape: (batch_size, )]
    total_question_aggregate_scores: Optional[SequenceScores]  # Shape: (batch_size, )]
    max_question_aggregate_scores: Optional[SequenceScores]  # Shape: (batch_size, )]
    min_question_aggregate_scores: Optional[SequenceScores]  # Shape: (batch_size, )]
    
    # Accuracy scores
    accuracy_aggregate_scores: Optional[torch.FloatTensor]  # Shape: (batch_size, )]
    
    # Mean among correct scores
    mean_among_correct_aggregate_scores: Optional[TokenScores]  # Shape: (batch_size, )]
        
@dataclass
class DatasetMetrics:
    """Container for dataset-level metrics"""
    accuracy_score: Optional[float]  # Shape: (1, )]
    aucroc_score: Optional[float]  # Shape: (1, )]
    aucpr_score: Optional[float]  # Shape: (1, )]
    brier_score: Optional[float]  # Shape: (1, )]
    # logloss_score: Optional[float]  # Shape: (1, )]
    mean_scores: Optional[TokenScores]  # Shape: (1, )]
