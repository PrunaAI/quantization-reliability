import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

from src.reliability.constants import dataset_expert_mapping, dataset_example_questions

import logging
logger = logging.getLogger("quant_logger")

def get_prompt(query, strategy, dataset_name):
    if strategy == "Original":
        return query
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
    elif strategy == "Direct Query":
        return f"{query}?"
    elif strategy == "Factual Retrieval":
        return f"Based on known facts, what is the answer to the following: {query}?"
    elif strategy == "First Thought":
        return f"What is the first thing that comes to your mind when asked: {query}?"
    elif strategy == "Deductive Reasoning":
        return f"Given these facts: 1) Paris is the capital of France. 2) The Eiffel Tower is located in Paris. 3) French is the official language of France. Deduce the answer to the following: {query}."
    elif strategy == "Expert Persona":
        if dataset_name in dataset_expert_mapping:
            expert_type, expert_institution = dataset_expert_mapping[dataset_name].split(', ')
            return f"You are a professor of {expert_type} at {expert_institution}. One of your students asks you: {query}\nAs an expert in this field, your answer is:"
        else:
            return f"As an expert in this field, please answer the following question: {query}"
    elif strategy == "Reflective Reasoning":
        return f"""You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries.
Follow these steps:

1. Think through the problem step by step within the <thinking> tags.
2. Reflect on your thinking to check for any errors or improvements within the <reflection> tags.
3. Make any necessary adjustments based on your reflection.
4. Provide your final, concise answer within the <output> tags.

Important: The <thinking> and <reflection> sections are for your internal reasoning process only.
Do not include any part of the final answer in these sections.
The actual response to the query must be entirely contained within the <output> tags.

Use the following format for your response:
<thinking>
[Your step-by-step reasoning goes here. This is your internal thought process, not the final answer.]
</thinking>
<reflection>
[Your reflection on your reasoning, checking for errors or improvements]
</reflection>
[Any adjustments to your thinking based on your reflection]
<output>
[Your final, concise answer to the query. This is the only part that will be shown to the user.]
</output>

Now, please answer the following question:
{query}. Answer:"""
    elif strategy == "Zero-Shot":
        return f"Q: {query}\nA:"
    elif strategy == "One-Shot":
        examples = dataset_example_questions[dataset_name]
        return f"Q: {examples[0][0]}\nA: {examples[0][1]}\n\nQ: {query}\nA:"
    elif strategy == "Two-Shot":
        examples = dataset_example_questions[dataset_name]
        return f"Q: {examples[0][0]}\nA: {examples[0][1]}\n\nQ: {examples[1][0]}\nA: {examples[1][1]}\n\nQ: {query}\nA:"
    elif strategy == "Three-Shot":
        examples = dataset_example_questions[dataset_name]
        return f"Q: {examples[0][0]}\nA: {examples[0][1]}\n\nQ: {examples[1][0]}\nA: {examples[1][1]}\n\nQ: {examples[2][0]}\nA: {examples[2][1]}\n\nQ: {query}\nA:"
    elif strategy == "Four-Shot":
        examples = dataset_example_questions[dataset_name]
        return f"Q: {examples[0][0]}\nA: {examples[0][1]}\n\nQ: {examples[1][0]}\nA: {examples[1][1]}\n\nQ: {examples[2][0]}\nA: {examples[2][1]}\n\nQ: {examples[3][0]}\nA: {examples[3][1]}\n\nQ: {query}\nA:"
    elif strategy == "Five-Shot":
        examples = dataset_example_questions[dataset_name]
        return f"Q: {examples[0][0]}\nA: {examples[0][1]}\n\nQ: {examples[1][0]}\nA: {examples[1][1]}\n\nQ: {examples[2][0]}\nA: {examples[2][1]}\n\nQ: {examples[3][0]}\nA: {examples[3][1]}\n\nQ: {examples[4][0]}\nA: {examples[4][1]}\n\nQ: {query}\nA:"
    else:
        return query

def calculate_entropy(probs):
    return -np.sum(probs * np.log(probs))

def plot_precision_recall_curve(y_true, y_pred, save_path):
    """
    Plot and save the Precision-Recall curve.
    
    Args:
        y_true: Array of ground truth labels
        y_pred: Array of predicted probabilities
        save_path: Path to save the plot
    """
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred)
    auc_score = metrics.average_precision_score(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', label=f'AUCPR = {auc_score:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    logger.info(f"Saved AUCPR plot to {save_path}")