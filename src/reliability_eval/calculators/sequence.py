from typing import Dict, List, Callable
from enum import Enum
import numpy as np
import torch
from evaluate import load
from rapidfuzz import fuzz
import wandb
from src.loggers.wandb_utils import safe_wandb_log

from src.reliability_eval.calculators.base import ScoresCalculator
from src.reliability_eval.common.score_types import SequenceScoreTypes
from src.reliability_eval.common.scores import SequenceAggregatedScores, SequenceScores, TokenScores

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class AnswerChecker:
    def __init__(self):
        # Load metrics once during initialization
        self.rouge_metric = load("rouge")
        
    def check_answers_batch(
        self, 
        true_answers: List[str], 
        relevant_texts: List[str], 
        full_texts: List[str],
        dataset_name: str, 
        num_shots: int = 0
    ) -> List[bool]:
        """Batch process correctness checks for all strategies"""
        dataset_name = dataset_name.lower()

        # Convert numpy arrays to Python scalars, handling multi-dimensional arrays
        true_answers = [
            ans.item() if isinstance(ans, np.ndarray) and ans.size == 1
            else ans.tolist() if isinstance(ans, np.ndarray)
            else ans for ans in true_answers
        ]
        relevant_texts = [
            text.item() if isinstance(text, np.ndarray) and text.size == 1
            else text.tolist() if isinstance(text, np.ndarray)
            else text for text in relevant_texts
        ]
        full_texts = [
            text.item() if isinstance(text, np.ndarray) and text.size == 1
            else text.tolist() if isinstance(text, np.ndarray)
            else text for text in full_texts
        ]
        
        safe_wandb_log({
            "true_answers": true_answers,
            "relevant_texts": relevant_texts,
            "full_texts": full_texts
        })
        
        # Rest of the method remains unchanged
        if dataset_name == "fktc":
            safe_wandb_log({"evaluation": "mixed"})
            return self._check_mixed_batch(
                true_answers,
                relevant_texts=relevant_texts,
                full_texts=full_texts
            )
        elif dataset_name == "commonsenseqa":
            if num_shots > 0:
                safe_wandb_log({"evaluation": "few-shot"})
                return self._check_few_shot_batch(
                    true_answers=true_answers,
                    relevant_texts=relevant_texts
                )
            safe_wandb_log({"evaluation": "exact"})
            return self._check_exact_batch(
                true_answers=true_answers,
                relevant_texts=relevant_texts
            )
        elif dataset_name == "triviaqa":
            safe_wandb_log({"evaluation": "rouge"})
            return self._check_mixed_batch(
                true_answers=true_answers,
                relevant_texts=relevant_texts,
                full_texts=full_texts
            )
        elif dataset_name == "coqa":
            safe_wandb_log({"evaluation": "rouge"})
            return self._check_mixed_batch(
                true_answers=true_answers,
                relevant_texts=relevant_texts,
                full_texts=full_texts
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _check_mixed_batch(
        self,
        true_answers: List[str],
        relevant_texts: List[str],
        full_texts: List[str]
    ) -> List[bool]:
        """Batch process mixed strategy checks"""
        # Batch compute ROUGE-L
        rouge_results = self.rouge_metric.compute(
            predictions=relevant_texts,
            references=true_answers,
            use_aggregator=False
        )['rougeL']
        
        def _check_answer_fuzzy(true_answer: str, generated_text: str, threshold: int = 80) -> bool:        
            generated_words = generated_text.lower().split()
            reference_words = true_answer.lower().split()
            n_ref_words = len(reference_words)
            
            # If single word reference, check each word separately
            if n_ref_words == 1:
                for word in generated_words:
                    if fuzz.ratio(word, reference_words[0]) >= threshold:
                        return True
                return False
            
            # If multi-word reference, check consecutive word groups
            for i in range(len(generated_words) - n_ref_words + 1):
                word_chunk = ' '.join(generated_words[i:i + n_ref_words])
                if fuzz.ratio(word_chunk, true_answer.lower()) >= threshold:
                    return True
            return False
        
        # Batch compute fuzzy matches using RapidFuzz
        fuzzy_results = [
            _check_answer_fuzzy(
                true_answer=true,
                generated_text=text
            )
            for true, text in zip(true_answers, full_texts)
        ]
        
        return [
            rouge > 0.3 or fuzzy
            for rouge, fuzzy in zip(rouge_results, fuzzy_results)
        ]

    def _check_exact_batch(self, true_answers: List[str], relevant_texts: List[str]) -> List[bool]:
        """Batch exact match checking"""
        return [
            true in text
            for true, text in zip(true_answers, relevant_texts)
        ]

    def _check_few_shot_batch(self, true_answers: List[str], relevant_texts: List[str]) -> List[bool]:
        """Batch few-shot completion checking"""
        templates = [
            f"So the answer is {true}."
            for true in true_answers
        ]
        return [
            template.lower() in text.lower()  # Case-insensitive comparison
            for template, text in zip(templates, relevant_texts)
        ]

    def _check_rouge_batch(self, true_answers: List[str], relevant_texts: List[str], threshold: float = 0.3) -> List[bool]:
        """Batch ROUGE-L checking"""
        rouge_scores = self.rouge_metric.compute(
            predictions=relevant_texts,
            references=true_answers,
            use_aggregator=False
        )['rougeL']
        
        return [score > threshold for score in rouge_scores]

class SequenceScoresCalculator(ScoresCalculator):
    def __init__(
        self,
        model,
        tokenizer,
        inference_config: Dict = {},
    ):
        """Initialize calculator with tokenizer and generation settings"""
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            inference_config=inference_config
        )
        
    def load_entailment_model(self, device="cuda"):
        """
        Loads a DeBERTa model fine-tuned on MNLI for textual entailment.
        
        Args:
            device (str): Device to load the model on ('cuda' or 'cpu')
        
        Returns:
            model: The loaded model
            tokenizer: The associated tokenizer
        """
        print("Loading DeBERTa model for semantic clustering...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli", cache_dir="cache")
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli", cache_dir="cache").to(device)
        return model, tokenizer

    def check_bidirectional_entailment(self, context, text1, text2, model, tokenizer, device="cuda"):
        """
        Checks if two texts entail each other bidirectionally in the given context.
        
        Args:
            context (str): The context (e.g., question) in which to evaluate equivalence
            text1 (str): First text (answer)
            text2 (str): Second text (answer)
            model: The entailment model
            tokenizer: The tokenizer for the model
            device (str): Device to run inference on
        
        Returns:
            bool: True if the texts are semantically equivalent, False otherwise
        """
        # Format inputs exactly like the reference implementation
        qa_1 = context + ' ' + text1
        qa_2 = context + ' ' + text2
        
        # Check forward entailment
        input_forward = qa_1 + ' [SEP] ' + qa_2
        encoded_input = tokenizer.encode(input_forward, padding=True)
        with torch.no_grad():
            forward_prediction = model(torch.tensor([encoded_input], device=device))['logits']
        forward_label = torch.argmax(forward_prediction, dim=1)
        
        # Check backward entailment
        input_backward = qa_2 + ' [SEP] ' + qa_1
        encoded_reverse_input = tokenizer.encode(input_backward, padding=True)
        with torch.no_grad():
            backward_prediction = model(torch.tensor([encoded_reverse_input], device=device))['logits']
        backward_label = torch.argmax(backward_prediction, dim=1)

        return not (0 in forward_label or 0 in backward_label)

    def cluster_generations(self, context, generations, model, tokenizer, device="cuda"):
        """
        Clusters generated texts based on semantic equivalence using bidirectional entailment.
        
        Args:
            context (str): The context/question for the generations
            generations (list): List of generated text answers
            model: The entailment model
            tokenizer: The tokenizer for the model
            device (str): Device to run inference on
        
        Returns:
            dict: Maps each generation to its semantic cluster ID
            list: Semantic cluster IDs for each generation in the original order
        """
        n_generations = len(generations)
        semantic_set_ids = list(range(n_generations))  # Initially, each text is in its own cluster
        semantic_mapping = {text: i for i, text in enumerate(generations)}
        
        # Compare all pairs of generations
        for i in range(n_generations):
            for j in range(i + 1, n_generations):
                # Skip if already in the same cluster
                if semantic_set_ids[i] == semantic_set_ids[j]:
                    continue
                    
                # Check if semantically equivalent
                if self.check_bidirectional_entailment(context, generations[i], generations[j], model, tokenizer, device):
                    # Merge clusters: set all generations in cluster j to cluster i
                    old_cluster_id = semantic_set_ids[j]
                    new_cluster_id = semantic_set_ids[i]
                    
                    for k in range(n_generations):
                        if semantic_set_ids[k] == old_cluster_id:
                            semantic_set_ids[k] = new_cluster_id
                            semantic_mapping[generations[k]] = new_cluster_id
        
        # Normalize cluster IDs to be consecutive integers starting from 0
        unique_clusters = sorted(list(set(semantic_set_ids)))
        cluster_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
        
        normalized_semantic_set_ids = [cluster_id_mapping[cluster_id] for cluster_id in semantic_set_ids]
        normalized_semantic_mapping = {text: cluster_id_mapping[cluster_id] for text, cluster_id in semantic_mapping.items()}
        
        return normalized_semantic_mapping, normalized_semantic_set_ids
        
    def _get_first_non_none_token_scores(self, sequence_aggregate_scores: SequenceAggregatedScores) -> TokenScores:
        """Get the first non-None token scores from sequence aggregate scores"""
        for field in [
            sequence_aggregate_scores.mean_sequence_aggregate_scores,
            sequence_aggregate_scores.total_sequence_aggregate_scores,
            sequence_aggregate_scores.max_sequence_aggregate_scores,
            sequence_aggregate_scores.min_sequence_aggregate_scores
        ]:
            if field is not None:
                return field
        raise ValueError("All token scores fields are None in sequence_aggregate_scores")
        
    def calculate_sequence_scores(
        self,
        sequence_aggregate_scores: SequenceAggregatedScores,
        sequence_score_types: List[SequenceScoreTypes],
        true_answers: np.ndarray,
        queries: List[str] = None,  # Add this parameter to pass queries
    ) -> SequenceScores:
        """Calculate metrics for each sequence sequence"""
        
        sequence_aggregate_scores = sequence_aggregate_scores if SequenceScoreTypes.ID in sequence_score_types else None
        
        # Use helper method to get token scores
        token_scores = self._get_first_non_none_token_scores(sequence_aggregate_scores)

        is_correct = self._calculate_is_correct(
            token_scores=token_scores,
            true_answers=true_answers
        ) if SequenceScoreTypes.IS_CORRECT in sequence_score_types else None
        
        # # Calculate semantic entropy if queries are provided
        # if queries is not None and token_scores.negative_log_likelihood_scores is not None and token_scores.full_decoded_text is not None:
        #     # Assuming one query per batch item for simplicity
        #     semantic_entropy_scores = torch.zeros(len(queries))
            
        #     for i, query in enumerate(queries):
        #         # Get the subset of token_scores for this query
        #         query_token_scores = TokenScores(
        #             negative_log_likelihood_scores=token_scores.negative_log_likelihood_scores[i:i+1],
        #             full_decoded_text=[token_scores.full_decoded_text[i]],
        #             effective_lengths=token_scores.effective_lengths[i:i+1] if token_scores.effective_lengths is not None else None,
        #             log_likelihood_raw_scores=token_scores.log_likelihood_raw_scores[i:i+1] if token_scores.log_likelihood_raw_scores is not None else None,
        #         )
                
        #         # Calculate semantic entropy for this query
        #         semantic_entropy_scores[i] = self.calculate_semantic_entropy(query_token_scores, query)[0]
            
        #     # Store semantic entropy scores in token_scores
        #     token_scores.semantic_entropy_scores = semantic_entropy_scores
            
        token_scores.semantic_entropy_scores = None  # Placeholder if not calculated
        
        sequence_scores = SequenceScores(
            mean_sequence_aggregate_scores=sequence_aggregate_scores.mean_sequence_aggregate_scores,
            total_sequence_aggregate_scores=sequence_aggregate_scores.total_sequence_aggregate_scores,
            max_sequence_aggregate_scores=sequence_aggregate_scores.max_sequence_aggregate_scores,
            min_sequence_aggregate_scores=sequence_aggregate_scores.min_sequence_aggregate_scores,
            is_correct=is_correct
        )

        return sequence_scores
            
    def _calculate_is_correct(
        self,
        token_scores,
        true_answers
    ):
        if not hasattr(token_scores, 'relevant_decoded_text'):
            raise ValueError("TokenScores must contain relevant_decoded_text field")

        # Flatten all texts and answers
        batch_size = len(token_scores.relevant_decoded_text)
        num_outputs = len(token_scores.relevant_decoded_text[0])
        
        flat_true = []
        flat_relevant = []
        flat_full = []
        for b in range(batch_size):
            # Handle multi-dimensional true_answers by ensuring we have a scalar or list
            if isinstance(true_answers[b], np.ndarray):
                if true_answers[b].size == 1:
                    answer = true_answers[b].item()
                else:
                    # For arrays with multiple values, use the first one or join them
                    # depending on your requirements
                    answer = true_answers[b].tolist()
                    if isinstance(answer, list):
                        answer = answer[0]  # Or use " ".join(answer) if you want to combine them
            else:
                answer = true_answers[b]
                
            flat_true.extend([answer] * num_outputs)
            flat_relevant.extend(token_scores.relevant_decoded_text[b])
            flat_full.extend(token_scores.full_decoded_text[b])
        
        # Batch check all answers
        checker = AnswerChecker()
        is_correct_flat = checker.check_answers_batch(
            flat_true, 
            flat_relevant,
            flat_full,
            self.inference_config.get("dataset_name"),
            self.inference_config.get("num_shots", 0)
        )
        
        # Reshape to original dimensions
        return torch.tensor(is_correct_flat).view(batch_size, num_outputs)
    
    def calculate_semantic_entropy(
        self,
        token_scores: TokenScores,
        context_query: str,
        device: str = "cuda"
    ) -> torch.FloatTensor:
        """
        Calculate semantic entropy using bidirectional entailment clustering.
        
        Args:
            token_scores: TokenScores object with NLL scores and full_decoded_text
            context_query: The context/question for the generations
            device: Device to run inference on
        
        Returns:
            torch.FloatTensor: Semantic entropy scores [batch_size]
        """
        # Load the entailment model if not already loaded
        if not hasattr(self, 'entailment_model') or self.entailment_model is None:
            self.entailment_model, self.entailment_tokenizer = self.load_entailment_model(device)
        
        batch_size = token_scores.negative_log_likelihood_scores.shape[0]
        num_repeats = token_scores.negative_log_likelihood_scores.shape[1]
        
        # Log input shapes for debugging
        safe_wandb_log({
            "semantic_entropy_batch_size": batch_size,
            "semantic_entropy_num_repeats": num_repeats,
            "has_full_decoded_text": token_scores.full_decoded_text is not None,
            "has_nll_scores": token_scores.negative_log_likelihood_scores is not None,
            "has_effective_lengths": token_scores.effective_lengths is not None,
            "nll_scores_sequence_shape": str(token_scores.negative_log_likelihood_scores.shape) if token_scores.negative_log_likelihood_scores is not None else "None",
            "effective_lengths_shape": str(token_scores.effective_lengths.shape) if token_scores.effective_lengths is not None else "None"
        })
        
        # Log all available attributes in token_scores
        token_scores_keys = [attr for attr in dir(token_scores) if not attr.startswith('__')]
        token_scores_values = {key: getattr(token_scores, key) is not None for key in token_scores_keys if not callable(getattr(token_scores, key))}
        safe_wandb_log({"token_scores_attributes": token_scores_values})
        
        # Calculate sequence log probabilities from token NLLs
        sequence_log_probs = torch.zeros((batch_size, num_repeats))
        
        for batch_idx in range(batch_size):
            for repeat_idx in range(num_repeats):
                # Get NLL scores for this sequence
                token_nlls = -token_scores.log_likelihood_raw_scores[batch_idx, repeat_idx]
                
                # Log token NLL properties for debugging
                safe_wandb_log({
                    f"token_nlls_batch_{batch_idx}_repeat_{repeat_idx}_shape": list(token_nlls.shape) if hasattr(token_nlls, 'shape') else "scalar",
                    f"token_nlls_batch_{batch_idx}_repeat_{repeat_idx}_type": str(type(token_nlls)),
                    f"token_nlls_batch_{batch_idx}_repeat_{repeat_idx}_ndim": token_nlls.ndim if hasattr(token_nlls, 'ndim') else 0
                })
                
                # Get effective length for this sequence
                if token_scores.effective_lengths is not None:
                    effective_len = token_scores.effective_lengths[batch_idx, repeat_idx].item()
                    # Take sum of NLLs up to effective length
                    sequence_log_probs[batch_idx, repeat_idx] = -torch.sum(token_nlls[:effective_len])
                else:
                    # If no effective lengths, use all token NLLs
                    sequence_log_probs[batch_idx, repeat_idx] = -torch.sum(token_nlls)
        
        semantic_entropy_scores = torch.zeros(batch_size)
        
        # Process each query separately
        for i in range(batch_size):
            # Get the generated texts for this query
            texts = token_scores.full_decoded_text[i]
            
            # Cluster the generations by meaning
            _, semantic_set_ids = self.cluster_generations(context_query, texts, self.entailment_model, self.entailment_tokenizer, device)
            
            # Calculate log probabilities for each semantic cluster
            unique_clusters = torch.unique(torch.tensor(semantic_set_ids))
            cluster_log_probs = torch.zeros(len(unique_clusters))
            
            for j, cluster_id in enumerate(unique_clusters):
                # Find sequences that belong to this cluster
                cluster_mask = torch.tensor([idx == cluster_id for idx in semantic_set_ids])
                # Get log probabilities for this cluster and aggregate them with logsumexp
                aggregated_log_probs = sequence_log_probs[i][cluster_mask]
                # Store the aggregated log probability for this cluster
                cluster_log_probs[j] = torch.logsumexp(aggregated_log_probs, dim=0)

            # Apply constant shift (similar to llh_shift = 5.0)
            cluster_log_probs = cluster_log_probs - 5.0

            # Compute entropy using the formula from the original implementation
            semantic_entropy = -torch.sum(cluster_log_probs, dim=0) / torch.tensor(len(unique_clusters))
            semantic_entropy_scores[i] = semantic_entropy
        
        return semantic_entropy_scores
        
    def _calculate_correct_entropy(
        self,
        sequence_aggregate_scores: SequenceAggregatedScores,
        is_correct: List[bool]
    ) -> List[float]: 
        """Calculate entropy of correct sequences"""
        raise NotImplementedError("Correct entropy calculation not yet implemented.")

    def _calculate_answer_probability(
        self,
        token_scores: List[TokenScores],
        true_answer: str,
    ) -> float:
        """
        Calculate the probability of generating the true answer within the sequence.
        """
        raise NotImplementedError("Answer probability calculation not yet implemented.")