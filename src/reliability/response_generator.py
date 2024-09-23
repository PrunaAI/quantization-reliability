from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import pandas as pd
import numpy as np
import torch

from src.reliability.utils import calculate_entropy, clean_response, get_prompt

class ResponseGenerator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda", cache_dir="/nfs/students/daro/.cache/huggingface")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="cuda", cache_dir="/nfs/students/daro/.cache/huggingface")
        
        # Set pad_token_id to eos_token_id to avoid the warning
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()
        
    def clean_response(self, output_text, true_answer):
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

    def calculate_probabilities(self, token_probs, true_answer):
        probs = []
        cleaned_token_probs = []
        true_answer_tokens = self.tokenizer.encode(true_answer.lower(), add_special_tokens=False)
        true_answer_found = False
        
        for token, prob in token_probs:
            cleaned_token_probs.append((token, prob))
            probs.append(prob)
            
            if not true_answer_found and all(t in [self.tokenizer.decode([tid]).lower() for tid in true_answer_tokens] for t in cleaned_token_probs[-len(true_answer_tokens):]):
                true_answer_found = True
            
            if true_answer_found and '.' in token or '!' in token or '?' in token or '\n' in token:
                break
        
        probs = torch.tensor(probs)
        beam_prob = torch.exp(torch.sum(torch.log(probs))).item()
        beam_prob_adj = torch.exp((len(probs) ** -1) * torch.sum(torch.log(probs))).item()
        entropy = self.calculate_entropy(probs.numpy())
        
        return beam_prob, beam_prob_adj, entropy, cleaned_token_probs
    
    def generate_response(self, query, strategy, dataset_name, true_answer, max_new_tokens, temperature, use_beam_search, n_repeats=5, n_beams=5):
        prompt = get_prompt(query, strategy, dataset_name)
        inputs = self.tokenizer(prompt, return_tensors='pt').to("cuda")
        
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
            generation_config = {
                **generation_config,
                "num_beams": n_beams,
                "num_return_sequences": 1,
            }
            generation_config.pop("do_sample")
        
        results = []

        if use_beam_search:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],  # Provide attention_mask to avoid the warning
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

            for i in range(len(outputs.sequences)):
                output_text = self.tokenizer.decode(outputs.sequences[i], skip_special_tokens=True)
                output_text, cleaned = clean_response(query, output_text, strategy, dataset_name)

                token_probs = [(self.tokenizer.decode([outputs.sequences[i][j]]), round(trans_scores[i, j], 4))
                            for j in range(len(trans_scores[i]))]

                beam_prob = torch.exp(torch.sum(torch.log(torch.from_numpy(trans_scores[i])))).cpu().item()
                beam_prob_adj = torch.exp((len(trans_scores[0]) ** -1) * torch.sum(torch.log(torch.from_numpy(trans_scores[i])))).cpu().item()

                entropy = calculate_entropy(np.array(trans_scores[i]))
                is_correct = true_answer.lower() in output_text.lower()

                results.append({
                    "output_text": output_text,
                    "cleaned": cleaned,
                    "beam_prob": beam_prob,
                    "beam_prob_adj": beam_prob_adj,
                    "entropy": entropy,
                    "is_correct": is_correct,
                    "token_probs": token_probs,
                    "run": i + 1  # 1-indexed
                })

        else:
            for run in range(n_repeats):
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        generation_config=GenerationConfig(**generation_config),
                        max_new_tokens=max_new_tokens
                    )
                    
                    output_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                    cleaned_text, is_correct = self.clean_response(output_text, true_answer)
                    
                    token_probs = []
                    for pos_idx, beam_scores in enumerate(outputs.scores):
                        softmax_scores = torch.softmax(beam_scores, dim=-1)
                        token_id = outputs.sequences[0][pos_idx + len(inputs['input_ids'][0])]
                        token_prob = softmax_scores[0][token_id].item()
                        token_probs.append((self.tokenizer.decode([token_id]), round(token_prob, 4)))
                    
                    beam_prob, beam_prob_adj, entropy, cleaned_token_probs = self.calculate_probabilities(token_probs, true_answer)
                    
                    results.append({
                        "output_text": output_text,
                        "cleaned": cleaned_text,
                        "beam_prob": beam_prob,
                        "beam_prob_adj": beam_prob_adj,
                        "entropy": entropy,
                        "is_correct": is_correct,
                        "token_probs": cleaned_token_probs,
                        "run": run + 1
                    })

        return results
    