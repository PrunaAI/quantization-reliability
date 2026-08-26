import time
from typing import Dict, List

import torch
import numpy as np
import wandb
from src.loggers.wandb_utils import safe_wandb_log

import logging
from src.reliability_eval.calculators.base import ScoresCalculator
from src.reliability_eval.common.params import ExperimentConfigParam, GenerationConfigParam
from src.reliability_eval.common.score_types import TokenScoreTypes
from src.reliability_eval.common.scores import TokenScores
from src.model_loading.common.tokenizer_utils import get_actual_tokenizer


logger = logging.getLogger("reliability_eval")


class TokenScoresCalculator(ScoresCalculator):
    def __init__(
        self,
        model,
        tokenizer,
        inference_config: Dict = {},
        exp_id: str = None
    ):
        """Initialize calculator with tokenizer and generation settings"""
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            inference_config=inference_config,
        )
        self.exp_id = exp_id
        self.total_entries = 0
        # Extract actual tokenizer for VLM models
        self.actual_tokenizer = get_actual_tokenizer(tokenizer)
        
    def calculate_token_scores(
        self,
        outputs: torch.Tensor,
        true_answers: np.array,
        token_score_types: List[TokenScoreTypes],
    ) -> TokenScores:
        """Calculate sequence-level scores for each sequence"""
        token_scores = self._get_token_scores(
            outputs=outputs,
            true_answers=true_answers,
            token_score_types=token_score_types
        )
        
        return token_scores
        
    def _get_transition_scores(
        self,
        outputs: torch.Tensor,
    ) -> np.ndarray:
        """Calculate transition scores using model's compute_transition_scores"""
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Optional: force garbage collection
            import gc
            gc.collect()
        
        # Get scores with normalization and clip them
        start_time = time.time()
        transition_scores = self.model.compute_transition_scores(
            sequences=outputs.sequences,
            scores=outputs.scores,
            normalize_logits=True,  # This should be true to get probabilities
        ).cpu()
        compute_time = time.time() - start_time
        safe_wandb_log({"transition_scores_computation_time_seconds": compute_time})

        # Add clipping for numerical stability
        transition_scores = torch.clamp(transition_scores, min=-100, max=100)
        
        return transition_scores
        
    def _calculate_token_info(self, outputs):
        """Extract token information and decoded texts using raw logits."""
        logger.info("Extracting token information using raw logits")
        
        input_length = outputs.sequences.shape[1] - len(outputs.scores)
        batch_size = outputs.sequences.shape[0]
        num_return_sequences = self.inference_config.get(GenerationConfigParam.NUM_RETURN_SEQUENCES.value, 1)
        num_repeats = self.inference_config.get(ExperimentConfigParam.NUM_REPEATS.value, 1)
        batch_repeats_size = batch_size // (num_repeats * num_return_sequences)
        
        # Get generated tokens
        tokens = outputs.sequences[:, input_length:].cpu()
        
        # Initialize nested structure to hold token info and texts
        token_info_tuples = [[] for _ in range(batch_repeats_size)]
        full_decoded_texts = [[] for _ in range(batch_repeats_size)]
        relevant_decoded_texts = [[] for _ in range(batch_repeats_size)]
        
        for batch_idx in range(batch_size):
            real_batch_idx = batch_idx // (num_repeats * num_return_sequences)
            
            sequence_tuples = []
            token_sequence = tokens[batch_idx].tolist()
            
            for pos in range(len(token_sequence)):
                if pos >= len(outputs.scores):
                    break
                    
                token_id = token_sequence[pos]
                if token_id == self.actual_tokenizer.pad_token_id:  # Changed
                    break
                    
                log_probs = torch.log_softmax(outputs.logits[pos][batch_idx], dim=-1)
                prob = torch.exp(log_probs[token_id]).item()
                
                decoded_piece = self.actual_tokenizer.decode(token_id)  # Changed
                sequence_tuples.append((token_id, decoded_piece, round(prob, 3)))
            
            token_info_tuples[real_batch_idx].append(sequence_tuples)
            
            # Get full text
            try:
                end_idx = token_sequence.index(self.actual_tokenizer.eos_token_id)  # Changed
                full_text = self.actual_tokenizer.decode(token_sequence[:end_idx])  # Changed
            except ValueError:
                full_text = self.actual_tokenizer.decode(token_sequence)  # Changed
                
            full_decoded_texts[real_batch_idx].append(full_text)
            
            # Get relevant text
            stop_idx = min([full_text.find(c) for c in ['. ', '\n'] if full_text.find(c) != -1], default=len(full_text))
            relevant_text = full_text[:stop_idx]
            relevant_decoded_texts[real_batch_idx].append(relevant_text)
        
        safe_wandb_log({
            "token_info_count": sum(len(repeats) for repeats in token_info_tuples),
            "relevant_text_count": sum(len(repeats) for repeats in relevant_decoded_texts)
        })
        
        return token_info_tuples, full_decoded_texts, relevant_decoded_texts

    def _calculate_effective_lengths(self, outputs, max_new_tokens=None):
        """Calculate effective sequence lengths by finding the first pad token or EOS token."""
        logger.info("Calculating effective sequence lengths")
        
        batch_size = outputs.sequences.shape[0]
        input_length = outputs.sequences.shape[1] - len(outputs.scores)
        
        effective_lengths = torch.ones(batch_size, dtype=torch.long) * (max_new_tokens or len(outputs.scores))
        
        for batch_idx in range(batch_size):
            generated_tokens = outputs.sequences[batch_idx, input_length:].tolist()
            
            for i, token_id in enumerate(generated_tokens):
                if token_id == self.actual_tokenizer.pad_token_id or token_id == self.actual_tokenizer.eos_token_id:  # Changed
                    effective_lengths[batch_idx] = i
                    break
        
        safe_wandb_log({
            "effective_lengths_mean": effective_lengths.float().mean().item(),
            "effective_lengths_min": effective_lengths.min().item(),
            "effective_lengths_max": effective_lengths.max().item(),
            "effective_lengths_values": wandb.Histogram(effective_lengths.numpy())
        })
        
        return effective_lengths

    def _calculate_padded_sequences(self, outputs, max_new_tokens=None):
        """Extract and pad the generated sequences to have consistent shape."""
        logger.info("Extracting padded output sequences")
        
        batch_size = outputs.sequences.shape[0]
        input_length = outputs.sequences.shape[1] - len(outputs.scores)
        max_length = max_new_tokens or len(outputs.scores)
        
        padded_sequences = torch.ones(batch_size, max_length, dtype=torch.long) * self.actual_tokenizer.pad_token_id  # Changed
        
        for batch_idx in range(batch_size):
            generated_tokens = outputs.sequences[batch_idx, input_length:].tolist()
            actual_length = min(len(generated_tokens), max_length)
            padded_sequences[batch_idx, :actual_length] = torch.tensor(
                generated_tokens[:actual_length], dtype=torch.long
            )
        
        safe_wandb_log({"padded_sequences_shape": list(padded_sequences.shape)})
        
        return padded_sequences

    def _calculate_nll_scores(self, outputs, effective_lengths=None):
        """Calculate negative log-likelihood scores using log_softmax on raw logits."""
        logger.info("Calculating NLL scores using log_softmax")
        
        batch_size = outputs.sequences.shape[0]
        num_return_sequences = self.inference_config.get(GenerationConfigParam.NUM_RETURN_SEQUENCES.value, 1)
        num_repeats = self.inference_config.get(ExperimentConfigParam.NUM_REPEATS.value, 1)
        batch_repeats_size = batch_size // (num_repeats * num_return_sequences)
        max_new_tokens = self.inference_config.get(GenerationConfigParam.MAX_NEW_TOKENS.value)
        input_length = outputs.sequences.shape[1] - len(outputs.scores)
        
        # Initialize tensor to hold NLL scores with (batch_size, num_repeats, max_new_tokens) shape
        nll_scores = torch.zeros(batch_repeats_size, num_repeats * num_return_sequences, max_new_tokens)
        
        # For each position in each sequence
        for batch_idx in range(batch_size):
            real_batch_idx = batch_idx // (num_repeats * num_return_sequences)
            repeat_idx = batch_idx % (num_repeats * num_return_sequences)
            
            # Determine how many tokens to process for this batch item
            length_to_process = effective_lengths[batch_idx].item() if effective_lengths is not None else max_new_tokens
            length_to_process = min(length_to_process, len(outputs.logits))
            
            for pos in range(length_to_process):
                # Get the actual generated token at this position
                token_idx = input_length + pos
                if token_idx < outputs.sequences.shape[1]:
                    generated_token = outputs.sequences[batch_idx, token_idx].item()
                    
                    # Compute log_softmax over the vocabulary
                    log_probs = torch.log_softmax(outputs.logits[pos][batch_idx], dim=-1)
                    
                    # Get the log probability of the generated token
                    token_log_prob = log_probs[generated_token].item()
                    
                    # Store the negative log likelihood
                    nll_scores[real_batch_idx, repeat_idx, pos] = -token_log_prob
                    
        # Replace NaN values with default value of 17
        nll_scores = torch.nan_to_num(nll_scores, nan=17.0)
        
        safe_wandb_log({
            "nll_scores_shape": list(nll_scores.shape),
            "nll_scores_mean": nll_scores.mean().item(),
            "nll_scores_max": nll_scores.max().item(),
            "nll_scores_min": nll_scores.min().item(),
        })
        
        return nll_scores

    def _calculate_entropy_scores(self, outputs, effective_lengths=None):
        """Calculate entropy scores using log_softmax."""
        logger.info("Calculating entropy scores using log_softmax")
        
        batch_size = outputs.sequences.shape[0]
        num_return_sequences = self.inference_config.get(GenerationConfigParam.NUM_RETURN_SEQUENCES.value, 1)
        num_repeats = self.inference_config.get(ExperimentConfigParam.NUM_REPEATS.value, 1)
        batch_repeats_size = batch_size // (num_repeats * num_return_sequences)
        max_new_tokens = self.inference_config.get(GenerationConfigParam.MAX_NEW_TOKENS.value)
        top_k = 50
        
        # Initialize tensor to hold entropy scores with (batch_size, num_repeats, max_new_tokens) shape
        entropy_scores = torch.zeros(batch_repeats_size, num_repeats * num_return_sequences, max_new_tokens)
        
        for pos, score in enumerate(outputs.scores):
            score = score.cpu()
            log_probs = torch.log_softmax(score, dim=-1)
            probs = torch.exp(log_probs)
            top_log_probs, _ = torch.topk(log_probs, k=min(top_k, probs.size(-1)), dim=-1)
            top_probs = torch.exp(top_log_probs)
            position_entropy = -torch.sum(top_probs * top_log_probs, dim=-1)
            
            # Reshape and assign values
            for batch_idx in range(batch_size):
                real_batch_idx = batch_idx // (num_repeats * num_return_sequences)
                repeat_idx = batch_idx % (num_repeats * num_return_sequences)
                
                # Only store entropy for positions up to the effective length for each batch item
                if effective_lengths is None or pos < effective_lengths[batch_idx].item():
                    entropy_scores[real_batch_idx, repeat_idx, pos] = position_entropy[batch_idx]
        
        # Replace NaN values with default value of 17
        entropy_scores = torch.nan_to_num(entropy_scores, nan=17.0)
        
        safe_wandb_log({
            "entropy_scores_shape": list(entropy_scores.shape),
            "entropy_scores_mean": entropy_scores.mean().item(),
            "entropy_scores_max": entropy_scores.max().item(),
            "entropy_scores_min": entropy_scores.min().item(),
            "entropy_values": wandb.Histogram(entropy_scores.flatten().numpy())
        })
        
        return entropy_scores

    def _calculate_top_k_concentration(self, outputs, k=3, effective_lengths=None, epsilon=1e-10):
        """Calculate top-k concentration scores using log_softmax."""
        logger.info(f"Calculating top-{k} concentration scores using log_softmax")
        
        batch_size = outputs.sequences.shape[0]
        num_return_sequences = self.inference_config.get(GenerationConfigParam.NUM_RETURN_SEQUENCES.value, 1)
        num_repeats = self.inference_config.get(ExperimentConfigParam.NUM_REPEATS.value, 1)
        batch_repeats_size = batch_size // (num_repeats * num_return_sequences)
        max_new_tokens = self.inference_config.get(GenerationConfigParam.MAX_NEW_TOKENS.value)
        
        # Use configured k if available
        k = self.inference_config.get(ExperimentConfigParam.TOP_K_CONCENTRATION.value, k)
        
        # Initialize tensor to hold concentration scores
        concentration_scores = torch.zeros(batch_repeats_size, num_repeats * num_return_sequences, max_new_tokens)
        
        for pos, score in enumerate(outputs.scores):
            score = score.cpu()
            log_probs = torch.log_softmax(score, dim=-1)
            # Get top-k log_probs first
            topk_log_probs = log_probs.topk(k, dim=-1).values
            # Then exponentiate to get probabilities
            topk_values = torch.exp(topk_log_probs)
            position_concentration = 1 + epsilon - topk_values.sum(dim=-1)
            
            # Reshape and assign values
            for batch_idx in range(batch_size):
                real_batch_idx = batch_idx // (num_repeats * num_return_sequences)
                repeat_idx = batch_idx % (num_repeats * num_return_sequences)
                
                # Only store concentration for positions up to the effective length for each batch item
                if effective_lengths is None or pos < effective_lengths[batch_idx].item():
                    concentration_scores[real_batch_idx, repeat_idx, pos] = position_concentration[batch_idx]
        
        # Replace NaN values with default value of 1
        concentration_scores = torch.nan_to_num(concentration_scores, nan=1.0)
        
        safe_wandb_log({
            "topk_concentration_scores_shape": list(concentration_scores.shape),
            "topk_concentration_mean": concentration_scores.mean().item(),
            "topk_concentration_min": concentration_scores.min().item(),
            "topk_concentration_max": concentration_scores.max().item()
        })
        
        return concentration_scores

    def _calculate_cross_entropy_scores(self, outputs, true_answers, effective_lengths=None, epsilon=1e-10):
        """Calculate cross entropy between model distribution and ground truth tokens using log_softmax."""
        logger.info("Calculating cross entropy scores using log_softmax")
        batch_size = outputs.sequences.shape[0]
        num_return_sequences = self.inference_config.get(GenerationConfigParam.NUM_RETURN_SEQUENCES.value, 1)
        num_repeats = self.inference_config.get(ExperimentConfigParam.NUM_REPEATS.value, 1)
        batch_repeats_size = batch_size // (num_repeats * num_return_sequences)
        max_new_tokens = self.inference_config.get(GenerationConfigParam.MAX_NEW_TOKENS.value)
        
        # Initialize tensor to hold cross entropy scores
        cross_entropy_scores = torch.zeros(batch_repeats_size, num_repeats * num_return_sequences, max_new_tokens)
        
        # Generate answer variations and tokenize them
        answer_variations = self._generate_answer_variations(true_answers)
        tokenized_variations = self._tokenize_answer_variations(answer_variations)
        
        # Calculate cross entropy for each position
        for batch_idx in range(batch_size):
            real_batch_idx = batch_idx // (num_repeats * num_return_sequences)
            repeat_idx = batch_idx % (num_repeats * num_return_sequences)
            
            # Determine how many tokens to process for this batch item
            length_to_process = effective_lengths[batch_idx].item() if effective_lengths is not None else max_new_tokens
            length_to_process = min(length_to_process, len(outputs.logits))
            
            # Compute log_softmax over the vocabulary
            log_probs = [torch.log_softmax(outputs.logits[pos_idx][batch_idx].cpu(), dim=-1) for pos_idx in range(length_to_process)]
            
            # Only compute actual values up to the effective length
            for token_pos in range(length_to_process):
                variation_scores_all = []
                for repetition_idx, repetition_token_seq in enumerate(tokenized_variations[real_batch_idx]):
                    # Collect scores for each variation
                    variation_scores = []
                    for token_seq_idx, token_seq in enumerate(repetition_token_seq):
                        var_token_scores = []
                        for true_answer_token_pos, true_answer_token in enumerate(token_seq):
                            if token_pos + true_answer_token_pos < length_to_process:
                                # Get log probability of correct token at this position
                                combined_pos = token_pos + true_answer_token_pos
                                try:
                                    true_answer_token_idx = true_answer_token.long() if isinstance(true_answer_token, torch.Tensor) else torch.tensor(true_answer_token, dtype=torch.long)
                                    token_log_prob = log_probs[combined_pos][true_answer_token_idx].item()
                                except (IndexError, TypeError, ValueError):
                                    token_log_prob = -100  # Set to a very low log probability when index error occurs
                                var_token_scores.append(token_log_prob)
                            else:
                                var_token_scores.append(-100)  # Placeholder for padding tokens
                        
                        # Sum log probs to get cumulative probability for this variation
                        var_score_sum = sum(var_token_scores)
                        variation_scores.append(var_score_sum)
                    
                    variation_scores_all.extend(variation_scores)
                
                # Take the maximum probability among variations if any exist
                if variation_scores_all:
                    max_score = max(variation_scores_all)
                    cross_entropy_scores[real_batch_idx, repeat_idx, token_pos] = -max_score
        
        # Replace NaN values with default value of 17
        cross_entropy_scores = torch.nan_to_num(cross_entropy_scores, nan=17.0)
        
        safe_wandb_log({
            "cross_entropy_scores_shape": list(cross_entropy_scores.shape),
            "cross_entropy_mean": cross_entropy_scores.mean().item(),
            "cross_entropy_min": cross_entropy_scores.min().item(),
            "cross_entropy_max": cross_entropy_scores.max().item()
        })
        
        return cross_entropy_scores

    def _get_token_scores(
        self,
        outputs,
        true_answers: np.array,
        token_score_types: List[TokenScoreTypes]
    ) -> TokenScores:
        """Convert model outputs to TokenScores objects with improved calculation methods."""
        logger.info("Getting token scores with improved calculation methods")
        start_time = time.time()
        
        # Basic dimensions
        batch_size = outputs.sequences.shape[0]
        num_return_sequences = self.inference_config.get(GenerationConfigParam.NUM_RETURN_SEQUENCES.value, 1)
        num_repeats = self.inference_config.get(ExperimentConfigParam.NUM_REPEATS.value, 1)
        batch_repeats_size = batch_size // (num_repeats * num_return_sequences)
        max_new_tokens = self.inference_config.get(GenerationConfigParam.MAX_NEW_TOKENS.value)
        
        # Log basic info
        safe_wandb_log({
            "batch_size": batch_size,
            "num_return_sequences": num_return_sequences,
            "num_repeats": num_repeats,
            "batch_repeats_size": batch_repeats_size,
            "max_new_tokens": max_new_tokens
        })
        
        # Calculate token info tuples, full texts, effective lengths, and padded sequences
        token_info_tuples, full_decoded_texts, relevant_decoded_texts = self._calculate_token_info(outputs)
        effective_lengths = self._calculate_effective_lengths(outputs, max_new_tokens=max_new_tokens)
        padded_sequences = self._calculate_padded_sequences(outputs, max_new_tokens=max_new_tokens).view(batch_repeats_size, num_repeats * num_return_sequences, max_new_tokens)
        
        # Calculate scores based on requested token score types
        nll_scores = None
        entropy_scores = None
        top_k_concentration_scores = None
        cross_entropy_confidence_scores = None
        
        # Calculate NLL scores
        start_time = time.time()
        nll_scores = self._calculate_nll_scores(
            outputs,
            effective_lengths=effective_lengths
        )
        nll_time = time.time() - start_time
        safe_wandb_log({"token_nll_calculation_time_seconds": nll_time})
        
        # Calculate entropy scores
        if TokenScoreTypes.ENTROPY in token_score_types:
            start_time = time.time()
            entropy_scores = self._calculate_entropy_scores(
                outputs,
                effective_lengths=effective_lengths
            )
            entropy_time = time.time() - start_time
            safe_wandb_log({"token_entropy_calculation_time_seconds": entropy_time})
        
        # Calculate top-k concentration scores
        if TokenScoreTypes.TOP_K in token_score_types:
            start_time = time.time()
            top_k_concentration_scores = self._calculate_top_k_concentration(
                outputs, 
                k=self.inference_config.get(ExperimentConfigParam.TOP_K_CONCENTRATION.value, 3),
                effective_lengths=effective_lengths
            )
            concentration_time = time.time() - start_time
            safe_wandb_log({"top_k_concentration_calculation_time_seconds": concentration_time})
        
        # Calculate cross-entropy scores
        if TokenScoreTypes.CROSS_ENTROPY in token_score_types:
            start_time = time.time()
            cross_entropy_confidence_scores = self._calculate_cross_entropy_scores(
                outputs, true_answers, effective_lengths=effective_lengths
            )
            cross_entropy_time = time.time() - start_time
            safe_wandb_log({"cross_entropy_calculation_time_seconds": cross_entropy_time})
        
        # Log processing time
        processing_time = time.time() - start_time
        safe_wandb_log({"token_scores_processing_time": processing_time})
        
        # Return TokenScores object with all calculated scores
        return TokenScores(
            token_info_tuples=token_info_tuples,
            full_decoded_text=full_decoded_texts,
            relevant_decoded_text=relevant_decoded_texts,
            cross_entropy_confidence_scores=cross_entropy_confidence_scores,
            negative_log_likelihood_scores=nll_scores,
            entropy_scores=entropy_scores,
            top_k_concentration_scores=top_k_concentration_scores,
            effective_lengths=effective_lengths.view(batch_repeats_size, num_repeats * num_return_sequences),
            padded_sequences=padded_sequences,
            log_likelihood_raw_scores=-nll_scores
        )
    
    def _generate_answer_variations(self, answers: np.array) -> List[List[str]]:
        """Generates valid string variations of answers with punctuation."""
        import string
        
        # Function to strip punctuation only from the start and end
        def strip_outer_punctuation(s):
            # Strip punctuation from start
            start_idx = 0
            while start_idx < len(s) and s[start_idx] in string.punctuation:
                start_idx += 1
            
            # If string is all punctuation, return empty string
            if start_idx >= len(s):
                return ""
            
            # Strip punctuation from end
            end_idx = len(s) - 1
            while end_idx >= 0 and s[end_idx] in string.punctuation:
                end_idx -= 1
            
            # Return the string with outer punctuation removed
            return s[start_idx:end_idx + 1]

        answer_variations = []
        for query_answers in answers:
            variations = []
            for answer in query_answers:
                # Remove punctuation only from the beginning and end of the answer
                answer = strip_outer_punctuation(answer)
                answer_vars = [
                    answer,                                     # 1) Original answer
                    " " + answer,                               # 2) Answer with space in front
                    answer[0].upper() + answer[1:] if answer else "",  # 3) First letter uppercase
                    " " + (answer[0].upper() + answer[1:] if answer else ""),  # 4) First letter uppercase with space
                    answer.title(),                             # 5) All words capitalized
                    " " + answer.title(),                       # 6) All words capitalized with space
                    answer.lower(),                             # 7) All lowercase
                    " " + answer.lower(),                       # 8) All lowercase with space
                    answer.replace(" ", "")                     # 9) No spaces
                ]
                # Remove duplicates
                answer_vars = list(dict.fromkeys(answer_vars))
                variations.append(answer_vars)
            answer_variations.append(variations)
        return answer_variations

    def _tokenize_answer_variations(self, answer_variations: List[List[str]]) -> List[List[torch.Tensor]]:
        """Converts answer string variations to token sequences."""
        tokenized_variations = []
        for query_answer_variations in answer_variations:
            tokenized_query_variations = []
            for repeated_answer_variations in query_answer_variations:
                tokenized_repeated_variations = []
                for variation in repeated_answer_variations:
                    tokens = self.actual_tokenizer.encode(  # Changed
                        variation, 
                        add_special_tokens=False,
                        return_tensors="pt"
                    )[0]
                    tokenized_repeated_variations.append(tokens)
                    
                    bos_token_id = self.actual_tokenizer.bos_token_id  # Changed
                    if bos_token_id is not None:
                        tokens_with_bos = torch.cat([torch.tensor([bos_token_id], device=tokens.device), tokens])
                    else:
                        tokens_with_bos = tokens
                    tokenized_repeated_variations.append(tokens_with_bos)
                
                tokenized_query_variations.append(tokenized_repeated_variations)
            tokenized_variations.append(tokenized_query_variations)
        
        return tokenized_variations
