from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import pandas as pd
import numpy as np
import torch


class ResponseGenerator:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda", cache_dir="/nfs/students/daro/.cache/huggingface")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="cuda", cache_dir="/nfs/students/daro/.cache/huggingface")
        
        # Set pad_token_id to eos_token_id to avoid the warning
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()

    def generate_response(self, query, strategy, true_answer, max_new_tokens, temperature, use_beam_search, n_repeats=5, n_beams=5):
        prompt = get_prompt(query, strategy)
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
                output_text = clean_response(output_text, strategy)

                token_probs = [(self.tokenizer.decode([outputs.sequences[i][j]]), round(trans_scores[i, j], 4))
                            for j in range(len(trans_scores[i]))]

                beam_prob = torch.exp(torch.sum(torch.log(torch.from_numpy(trans_scores[i])))).cpu().item()
                beam_prob_adj = torch.exp((len(trans_scores[0]) ** -1) * torch.sum(torch.log(torch.from_numpy(trans_scores[i])))).cpu().item()

                entropy = calculate_entropy(np.array(trans_scores[i]))
                is_correct = true_answer.lower() in output_text.lower()

                results.append({
                    "output_text": output_text,
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
                output_text = clean_response(output_text, strategy)

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
                    "beam_prob": beam_prob,
                    "beam_prob_adj": beam_prob_adj,
                    "entropy": entropy,
                    "is_correct": is_correct,
                    "token_probs": token_probs,
                    "run": run + 1  # 1-indexed
                })

        return results
    
def get_prompt(query, strategy):
    if strategy == "Fact Statement":
        return f"{query} Fact:"
    elif strategy == "Completion":
        return f"{query} The answer is:"
    elif strategy == "Definitive Statement":
        return f"The answer to the question '{query}' is:"
    elif strategy == "Fill-in-the-Blank":
        return f"{query} The answer is: _____.\nAnswer:"
    elif strategy == "Structured Answer Prompt":
        return f"Question: {query}\nAnswer (one word):"
    elif strategy == "Direct Instruction":
        return f"Please answer the following question in one word.\nQuestion: {query}\nAnswer:"
    elif strategy == "Contextual Prompts":
        return f"{query} (Please answer in one word)"
    elif strategy == "Question-Answer Pairs":
        return f"QSTN: What is the capital of France?\nANSR: Paris\nQSTN: What is the capital of Germany?\nANSR: Berlin\nQSTN: {query}\nANSR:"
    elif strategy == "Direct Answer":
        return f"Please provide a short, direct answer to the following question: {query} Answer:"
    elif strategy == "Q&A Format":
        return f"Q: {query}\nA:"
    elif strategy == "Instructional":
        return f"Answer the following question in one or two words: {query}"
    elif strategy == "Summary":
        return f"Summarize the answer to the following question: {query}"
    elif strategy == "Echo":
        return f"{query} {query}"
    elif strategy == "True Completion":
        return f"{query} The true answer is:"
    elif strategy == "Direct Completion":
        return f"{query} Answer:"
    elif strategy == "Answer Completion":
        return f"{query} The correct answer is:"
    
    # New Strategies
    elif strategy == "Direct Query":
        return f"{query}?"
    elif strategy == "Factual Retrieval":
        return f"Based on known facts, what is the answer to the following: {query}?"
    elif strategy == "First Thought":
        return f"What is the first thing that comes to your mind when asked: {query}?"
    elif strategy == "Deductive Reasoning":
        return f"Given these facts: 1) Paris is the capital of France. 2) The Eiffel Tower is located in Paris. 3) French is the official language of France. Deduce the answer to the following: {query}."
    else:
        return query

def clean_response(query, output_text, strategy):
    formatted_query = get_prompt(query, strategy)
    if output_text.startswith(formatted_query):
        output_text = output_text[len(formatted_query):]
        
    return output_text.strip()

def calculate_entropy(probs):
    return -np.sum(probs * np.log(probs))