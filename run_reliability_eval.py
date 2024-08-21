import shutil
import re
import os
import subprocess
import tempfile

import logging

logger = logging.getLogger("quant_logger")

logger.info("Setting up cache paths...")

# To avoid the following problem when running seml (see https://github.com/pytorch/pytorch/issues/37377)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# Disables parallelism to remove transformers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
CACHE_PATH = "/nfs/students/daro/.cache/huggingface"
HUB_PATH = os.path.join(CACHE_PATH, "hub")

if not os.path.exists(HUB_PATH):
    os.makedirs(HUB_PATH)
    print(f"Creating huggingface hub path at {HUB_PATH}")
    
print(f"Setting cache path to {CACHE_PATH}")
os.environ["TORCH_HOME"] = CACHE_PATH
os.environ["HF_HOME"] = CACHE_PATH
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_PATH
os.environ["HUGGINGFACE_ASSETS_CACHE"] = CACHE_PATH
os.environ["TRANSFORMERS_CACHE"] = CACHE_PATH

import time
import random
import torch

torch.hub.set_dir(CACHE_PATH)

# Empty the cache
import torch
with torch.no_grad():
    torch.cuda.empty_cache()
    
logging.info("Setting up working directory...")

#os.chdir('..')
logging.info(f"Current Working Directory: {os.getcwd()}")
import sys
sys.path.append("../") # Add directory containing src/data to path

import importlib
import src  # Assuming src is the package name

# Reload the src module after making changes
importlib.reload(src)

logging.info("Setting up working directory...")

# HuggingFace authentication
import os
from dotenv import load_dotenv
from huggingface_hub import login

logging.info("Authenticating Hugging Face...")

load_dotenv()
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
if huggingface_token is None:
    raise ValueError(
        f"Please set the HUGGINGFACE_TOKEN environment variable."
        f"Looking in {os.path.join(os.getcwd(), '.env')}"
    )
else:
    logging.info("Hugging Face token loaded successfully.")
login(token=huggingface_token, add_to_git_credential=True)

from seml.experiment import Experiment
import seml

logging.info("Setting up GPU memory usage list...")
# Global list to store GPU memory usage
gpu_memory_usage = {}

logging.info("Setting up SEML experiment...")

# Set the SEML experiment
ex = Experiment(save_git_info=False)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.automain
def run_reliability_eval():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score

    def get_prompt(query, strategy):
        if strategy == "Fact Statement":
            return f"{query} Fact:"
        elif strategy == "Completion":
            return f"{query} The answer is:"
        elif strategy == "Definitive Statement":
            return f"The answer to the question '{query}' is:"
        elif strategy == "Fill-in-the-Blank":
            return f"{query} is located in _____.\nAnswer:"
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
        else:
            return query

    def clean_response(output_text, strategy):
        if strategy == "Fact Statement":
            return output_text.split("Fact:")[-1].strip()
        elif strategy == "Completion":
            return output_text.split("The answer is:")[-1].strip()
        elif strategy == "Definitive Statement":
            return output_text.split("is:")[-1].strip()
        elif strategy == "Fill-in-the-Blank":
            return output_text.split("Answer:")[-1].strip()
        elif strategy == "Structured Answer Prompt":
            return output_text.split("Answer (one word):")[-1].strip()
        elif strategy == "Direct Instruction":
            return output_text.split("Answer:")[-1].strip()
        elif strategy == "Contextual Prompts":
            return output_text.split("Please answer in one word")[-1].strip()
        elif strategy == "Question-Answer Pairs":
            return output_text.split("ANSR:")[-1].strip()
        elif strategy == "Direct Answer":
            return output_text.split("Answer:")[-1].strip()
        elif strategy == "Q&A Format":
            return output_text.split("A:")[-1].strip()
        elif strategy == "Instructional":
            return output_text.split("Answer the following question")[-1].strip()
        elif strategy == "Summary":
            return output_text.split("Summarize the answer to the following question")[-1].strip()
        elif strategy == "Echo":
            return output_text
        elif strategy == "True Completion":
            return output_text.split("The true answer is:")[-1].strip()
        elif strategy == "Direct Completion":
            return output_text.split("Answer:")[-1].strip()
        elif strategy == "Answer Completion":
            return output_text.split("The correct answer is:")[-1].strip()
        else:
            return output_text.strip()

    def calculate_entropy(probs):
        return -np.sum(probs * np.log(probs))

    class ResponseGenerator:
        def __init__(self, model_name):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="cuda")
            
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
                    "num_return_sequences": n_beams,
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

                for i in range(n_beams):  # Process each beam
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

    # Example usage:
    model_names = [
        #"bigscience/bloomz-560m",
        "meta-llama/Meta-Llama-3-8B",
        #"meta-llama/Meta-Llama-3-8B-Instruct",
        "bigscience/bloomz-1b1",
        "openai-community/gpt2-large",
        #"EleutherAI/gpt-neo-1.3B",
        "TinyLlama/TinyLlama_v1.1",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]
    queries_and_answers = [
        # Place / Region Names (3)
        ("What is the location of the Great Wall of China?", "China"),
        ("Where is the Eiffel Tower located?", "Paris"),
        ("What is the capital city of Japan?", "Tokyo"),
        
        # Adjectives (3)
        ("How would you describe the taste of a lemon?", "Sour"),
        ("What is the best way to describe the weather on a clear day?", "Sunny"),
        ("How would you describe the feeling of touching velvet?", "Soft"),
        
        # Person Names (3)
        ("Who was the first president of the United States?", "George Washington"),
        ("Who is the author of 'Pride and Prejudice'?", "Jane Austen"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        
        # Abstract Answers (3)
        ("What is your favorite color?", "I don't know"),
        ("What do you think of the meaning of life?", "It's complicated"),
        ("What comes after the end?", "Nothing"),
        
        # Numerical Answers (3)
        ("How many continents are there on Earth?", "Seven"),
        ("What is the boiling point of water in Celsius?", "100"),
        ("How many hours are there in a day?", "24"),
        
        # Languages (3)
        ("What language is primarily spoken in Brazil?", "Portuguese"),
        ("Which language is spoken in Germany?", "German"),
        ("What is the official language of China?", "Mandarin"),
        
        # Company Names (3)
        ("Which company developed the iPhone?", "Apple"),
        ("What is the name of the online retailer founded by Jeff Bezos?", "Amazon"),
        ("Which company is known for its search engine?", "Google")
    ]

    # max_new_tokens_list = [10, 15, 20, 30, 40]
    max_new_tokens_list = [10, 15, 20, 30, 40]
    temperature_list = [0.1]
    strategies = [
        "Fact Statement",
        "Completion",
        "Definitive Statement",
        "Fill-in-the-Blank",
        "Structured Answer Prompt",
        "Direct Instruction",
        "Contextual Prompts",
        "Question-Answer Pairs",
        "Direct Answer",
        "Q&A Format",
        "Instructional",
        "Summary",
        "Echo",
        "True Completion",
        "Direct Completion",
        "Answer Completion"
    ]

    n_repeats = 5
    use_beam_search_list = [True, False]

    # Initialize a list to store the results
    results = []

    n_steps = 0
    total_steps = len(model_names) * len(queries_and_answers) * len(max_new_tokens_list) * len(temperature_list) * len(strategies) * n_repeats * len(use_beam_search_list)
    for model_name in model_names:
        generator = ResponseGenerator(model_name)
        for query_idx, (query, true_answer) in enumerate(queries_and_answers):
            for max_new_tokens in max_new_tokens_list:
                for temperature in temperature_list:
                    for strategy in strategies:
                        for use_beam_search in use_beam_search_list:  # Iterate over Beam Search and Multinomial Sampling
                            run_results = generator.generate_response(query, strategy, true_answer, max_new_tokens, temperature, use_beam_search, n_repeats=n_repeats, n_beams=n_repeats)
                            for result_dict in run_results:
                                print(f"TOTAL: {n_steps + 1}/{total_steps}, MODEL: {model_name}, QUERY: {query_idx}, STRATEGY: {strategy}, MAX_NEW_TOKENS: {max_new_tokens}, RUN: {result_dict['run']}/{n_repeats}")
                                # Store the results in the list
                                results.append({
                                    "Model": model_name,
                                    "Query": query,
                                    "Strategy": strategy,
                                    "Max New Tokens": max_new_tokens,
                                    "Temperature": temperature,
                                    "Beam Search": use_beam_search,
                                    "Run": result_dict['run'],
                                    "Generated Response": result_dict['output_text'],
                                    "P": result_dict['beam_prob'],
                                    "P_adj": result_dict['beam_prob_adj'],
                                    "Entropy": result_dict['entropy'],
                                    "Is Correct": result_dict['is_correct'],
                                    "Token Probabilities": result_dict['token_probs']
                                })
                                print(f"IS_CORRECT: {result_dict['is_correct']}, PROB: {result_dict['beam_prob']:.2f}, ADJ_PROB: {result_dict['beam_prob_adj']:.2f}, ENTROPY: {result_dict['entropy']:.2f}")
                                n_steps += 1
                                if n_steps >= 100:
                                    # pass
                                    raise KeyboardInterrupt
                                
    df_results = pd.DataFrame(results)
    df_results.to_excel("beam_multinomial_llama_08_21-raw_table.xlsx", index=False)

    df_results = pd.DataFrame(results)

    def calculate_scores(group):
        y_true = group['Is Correct'].values
        y_scores = group['P'].values
        
        # Ensure at least two classes are present for AUROC
        if len(set(y_true)) > 1:
            aucroc = roc_auc_score(y_true, y_scores)
        else:
            aucroc = np.nan
        
        aucpr = average_precision_score(y_true, y_scores)
        accuracy = np.mean(y_true)
        
        return pd.Series({'AUROC Score': aucroc, 'AUCPR Score': aucpr, 'Accuracy': accuracy})

    # Group by the relevant columns and calculate AUROC, AUCPR, and accuracy for each group
    df_scores = df_results.groupby(['Model', 'Max New Tokens', 'Temperature', 'Strategy', 'Beam Search']).apply(calculate_scores).reset_index()
    df_results = df_results.merge(df_scores, on=['Model', 'Max New Tokens', 'Temperature', 'Strategy', 'Beam Search'])

    # Save the original detailed results to an Excel file
    df_results.to_excel("beam_multinomial_llama_08_21-table.xlsx", index=False)
    df_scores.to_excel("beam_multinomial_llama_08_21-scores.xlsx", index=False)
    
    return df_results
