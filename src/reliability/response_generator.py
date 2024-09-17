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
                output_text, cleaned = clean_response(query, output_text, strategy, dataset_name)

                response_tokens = outputs.sequences[0].tolist()
                probs = []
                sequence = outputs.sequences[0]
                shift_idx = len(sequence) - len(outputs.scores)  # Position to start processing tokens
                token_probs = []

                for pos_idx, beam_scores in enumerate(outputs.scores):
                    softmax_scores = torch.softmax(beam_scores, dim=-1)
                    token_id = sequence[pos_idx + shift_idx]
                    token_prob = softmax_scores[0][token_id].item()
                    token_probs.append((self.tokenizer.decode([token_id]), round(token_prob, 4)))
                    probs.append(token_prob)

                probs = torch.tensor(probs)
                beam_prob = torch.exp(torch.sum(torch.log(probs))).item()
                beam_prob_adj = torch.exp((len(probs) ** -1) * torch.sum(torch.log(probs))).item()

                entropy = calculate_entropy(np.array(probs))
                is_correct = true_answer.lower() in output_text.lower()

                results.append({
                    "output_text": output_text,
                    "cleaned": cleaned,
                    "beam_prob": beam_prob,
                    "beam_prob_adj": beam_prob_adj,
                    "entropy": entropy,
                    "is_correct": is_correct,
                    "token_probs": token_probs,
                    "run": run + 1  # 1-indexed
                })

        return results
    