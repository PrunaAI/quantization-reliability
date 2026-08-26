from typing import Dict, List, Tuple

from src.reliability_eval.common.scores import QuestionAggregateScores, SequenceScores, TokenScores


def _group_scores_by_perturbations(scores: Dict[str, QuestionAggregateScores],
                                pert_types: List[str],
                                pert_intensities: List[int]) -> Dict[Tuple[str, int], Dict[str, QuestionAggregateScores]]:
    """Groups QuestionAggregateScores by perturbation type and intensity."""
    grouped_scores = {}
    
    for pert_type, pert_intensity in set(zip(pert_types, pert_intensities)):
        grouped_scores[(pert_type, pert_intensity)] = {}
        
        for pipeline_name, question_scores in scores.items():
            # Create new QuestionAggregateScores object for this group
            grouped_question_scores = QuestionAggregateScores(
                mean_question_aggregate_scores=None,
                total_question_aggregate_scores=None,
                max_question_aggregate_scores=None,
                min_question_aggregate_scores=None,
                accuracy_aggregate_scores=None,
                mean_among_correct_aggregate_scores=None
            )
            
            # Group accuracy scores if they exist
            if question_scores.accuracy_aggregate_scores is not None:
                mask = [(t == pert_type and i == pert_intensity) 
                        for t, i in zip(pert_types, pert_intensities)]
                grouped_question_scores.accuracy_aggregate_scores = \
                    question_scores.accuracy_aggregate_scores[mask]
            
            # Group sequence scores
            for aggregate_type in ['mean', 'total', 'max', 'min']:
                seq_scores = getattr(question_scores, f'{aggregate_type}_question_aggregate_scores')
                if seq_scores is None:
                    continue
                
                new_seq_scores = SequenceScores(
                    mean_sequence_aggregate_scores=None,
                    total_sequence_aggregate_scores=None,
                    max_sequence_aggregate_scores=None,
                    min_sequence_aggregate_scores=None,
                    is_correct=None
                )
                    
                # Group is_correct if it exists
                if seq_scores.is_correct is not None:
                    mask = [(t == pert_type and i == pert_intensity) 
                            for t, i in zip(pert_types, pert_intensities)]
                    new_seq_scores.is_correct = seq_scores.is_correct[mask]
                
                # Group token scores for each sequence aggregate
                for seq_aggregate_type in ['mean', 'total', 'max', 'min']:
                    token_scores = getattr(seq_scores, f'{seq_aggregate_type}_sequence_aggregate_scores')
                    if token_scores is None:
                        continue
                    
                    new_token_scores = TokenScores(
                        token_info_tuples=None,
                        full_decoded_text=None,
                        relevant_decoded_text=None,
                        effective_lengths=None,
                        padded_sequences=None,
                        cross_entropy_confidence_scores=None,
                        negative_log_likelihood_scores=None,
                        entropy_scores=None,
                        top_k_concentration_scores=None
                    )
                        
                    # Group each score type
                    for score_attr in ['cross_entropy_confidence_scores',
                                        'negative_log_likelihood_scores',
                                        'entropy_scores',
                                        'top_k_concentration_scores']:
                        scores_tensor = getattr(token_scores, score_attr)
                        if scores_tensor is None:
                            continue
                        
                        mask = [(t == pert_type and i == pert_intensity) 
                                for t, i in zip(pert_types, pert_intensities)]
                        setattr(new_token_scores, score_attr, scores_tensor[mask])
                        
                    setattr(new_seq_scores, f'{seq_aggregate_type}_sequence_aggregate_scores',
                            new_token_scores)
                
                setattr(grouped_question_scores, f'{aggregate_type}_question_aggregate_scores',
                        new_seq_scores)
            
            grouped_scores[(pert_type, pert_intensity)][pipeline_name] = grouped_question_scores
    
    return grouped_scores