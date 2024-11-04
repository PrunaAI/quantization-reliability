from transformers import GenerationConfig
import pandas as pd
import numpy as np
import torch

from src.reliability.utils import calculate_entropy, get_prompt
from src.models.load_model_and_tokenizer import load_model_and_tokenizer

class ResponseGenerator:
    def __init__(self, model_name, cache_dir):
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_name=model_name,
            device="cuda",
            max_memory=None,
            cache_dir=cache_dir
        )
        
        # Set pad_token_id to eos_token_id to avoid the warning
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()
        
    def clean_response(self, query, output_text, true_answer):
        # Strip the query from the beginning of the output if present
        if output_text.lower().startswith(query.lower()):
            output_text = output_text[len(query):].lstrip()
        
        output_lower = output_text.lower()
        true_answer_lower = true_answer.lower()
        
        start_index = output_lower.find(true_answer_lower)
        if start_index == -1:
            return output_text, False  # True answer not found
        
        end_index = start_index + len(true_answer_lower)
        anchor_index = next((i for i in range(end_index, len(output_text)) 
                             if output_text[i] in '.,!?;\n'), len(output_text))
        
        cleaned_text = output_text[:anchor_index].strip()
        return cleaned_text, True

    def calculate_probabilities(self, query, token_probs, true_answer):
        probs = []
        cleaned_token_probs = []
        true_answer_lower = true_answer.lower()
        true_answer_found = False
        query_tokens = self.tokenizer.encode(query.lower(), add_special_tokens=False)
        query_end_index = 0
        current_text = ""
        
        # Find the end of the query in the token list
        bof_adjustment_idx = 0
        for i, (token, _) in enumerate(token_probs):
            if i == 0 and token == '<|begin_of_text|>':
                query_end_index = i + 1
                bof_adjustment_idx = 1
            elif i < len(query_tokens) + bof_adjustment_idx and self.tokenizer.decode([query_tokens[i - bof_adjustment_idx]]).lower() in token.lower():
                query_end_index = i + 1
            else:
                break
        
        # Process tokens after the query
        for token, prob in token_probs[query_end_index:]:
            current_text += token
            
            if not true_answer_found and true_answer_lower in current_text.lower():
                true_answer_found = True

            if true_answer_found and any(char in token for char in '.,!?;:()[]{}"\'\n'):
                break

            cleaned_token_probs.append((token, prob))
            probs.append(prob)
        
        probs = torch.tensor(probs)
        beam_prob = torch.exp(torch.sum(torch.log(probs))).item() if len(probs) > 0 else 0
        beam_prob_adj = torch.exp((len(probs) ** -1) * torch.sum(torch.log(probs))).item() if len(probs) > 0 else 0
        entropy = calculate_entropy(probs.numpy()) if len(probs) > 0 else 0
        
        return beam_prob, beam_prob_adj, entropy, cleaned_token_probs
    
    def generate_responses(
        self,
        queries,
        strategy,
        dataset_name,
        true_answers,
        max_new_tokens,
        temperature,
        use_beam_search,
        n_repeats=5,
        n_beams=5,
        past_key_values=None
    ):
        prompts = [get_prompt(query, strategy, dataset_name) for query in queries]
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to("cuda")
        
        # Fix: Convert attention mask to boolean. Fixing error: "Expected attn_mask dtype to be bool or to match query dtype..."
        inputs['attention_mask'] = inputs['attention_mask'].bool()
        
        # Generation configuration
        generation_config = {
            "temperature": temperature,
            "do_sample": True,
            "top_p": 0.75,
            "top_k": 40,
            "output_scores": True,
            "output_hidden_states": False,
            "output_attentions": False,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        if use_beam_search:
            generation_config.update({
                "num_beams": n_beams,
                "num_return_sequences": 1,
            })
            generation_config.pop("do_sample")
        
        batch_results = []

        if use_beam_search:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    generation_config=GenerationConfig(**generation_config),
                    max_new_tokens=max_new_tokens
                )
            
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences,
                outputs.scores,
                normalize_logits=True,
                beam_indices=outputs.beam_indices
            )
            trans_scores = np.exp(transition_scores.cpu().numpy())

            for i, (query, true_answer) in enumerate(zip(queries, true_answers)):
                output_text = self.tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
                cleaned_text, is_correct = self.clean_response(query, output_text, true_answer)

                token_probs = [(self.tokenizer.decode([outputs.sequences[i][j + len(inputs['input_ids'][i])]]), round(trans_scores[i, j], 4))
                               for j in range(len(trans_scores[i]))]
                
                beam_prob, beam_prob_adj, entropy, cleaned_token_probs = self.calculate_probabilities(query, token_probs, true_answer)

                batch_results.append([{
                    "output_text": output_text,
                    "cleaned": cleaned_text,
                    "beam_prob": beam_prob,
                    "beam_prob_adj": beam_prob_adj,
                    "entropy": entropy,
                    "is_correct": is_correct,
                    "token_probs": cleaned_token_probs,
                    "run": 1  # Only one run for beam search
                }])

        else:
            batch_results = [[] for _ in queries]
            for run in range(n_repeats):
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        generation_config=GenerationConfig(**generation_config),
                        max_new_tokens=max_new_tokens
                    )
                    
                for i, (query, true_answer) in enumerate(zip(queries, true_answers)):
                    output_text = self.tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
                    cleaned_text, is_correct = self.clean_response(query, output_text, true_answer)
                    
                    token_probs = []
                    for pos_idx, beam_scores in enumerate(outputs.scores):
                        softmax_scores = torch.softmax(beam_scores, dim=-1)
                        token_id = outputs.sequences[i][pos_idx + len(inputs['input_ids'][i])]
                        token_prob = softmax_scores[i][token_id].item()
                        token_probs.append((self.tokenizer.decode([token_id]), round(token_prob, 4)))
                    
                    beam_prob, beam_prob_adj, entropy, cleaned_token_probs = self.calculate_probabilities(query, token_probs, true_answer)
                    
                    batch_results[i].append({
                        "output_text": output_text,
                        "cleaned": cleaned_text,
                        "beam_prob": beam_prob,
                        "beam_prob_adj": beam_prob_adj,
                        "entropy": entropy,
                        "is_correct": is_correct,
                        "token_probs": cleaned_token_probs,
                        "run": run + 1
                    })

        return batch_results
    