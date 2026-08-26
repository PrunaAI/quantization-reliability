from typing import Dict, Callable
from src.reliability_eval.prompting.types import PromptStrategy
from src.reliability_eval.common.constants import DATASET_TO_EXPERT_MAP, DATASET_EXAMPLE_QUERIES

def get_expert_persona_prompt(query: str, dataset_name: str) -> str:
    """Create expert persona prompt based on dataset."""
    if dataset_name in DATASET_TO_EXPERT_MAP:
        expert_type, expert_institution = DATASET_TO_EXPERT_MAP[dataset_name].split(', ')
        return f"You are a professor of {expert_type} at {expert_institution}. One of your students asks you: {query}\nAs an expert in this field, your answer is:"
    return f"As an expert in this field, please answer the following question: {query}"

def get_reflective_reasoning_prompt(query: str, dataset_name: str) -> str:
    """Create reflective reasoning prompt with chain-of-thought structure."""
    return f"""You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries.
Follow these steps:
1. Think through the problem step by step within the <thinking> tags.
2. Reflect on your thinking to check for any errors or improvements within the <reflection> tags.
3. Make any necessary adjustments based on your reflection.
4. Provide your final, concise answer within the <output> tags.
Now, please answer the following question: {query}. Answer:"""

class PromptStrategyFactory:
    """Factory for creating and applying prompt strategies."""

    STRATEGIES: Dict[PromptStrategy, Callable[[str, str], str]] = {
        PromptStrategy.ORIGINAL: lambda query, _: query,
        PromptStrategy.FACT_STATEMENT: lambda query, _: f"{query} Fact:",
        PromptStrategy.COMPLETION: lambda query, _: f"{query} The answer is:",
        PromptStrategy.DEFINITIVE_STATEMENT: lambda query, _: f"The answer to the question '{query}' is:",
        PromptStrategy.FILL_IN_BLANK: lambda query, _: f"{query} The answer is: _____.\nAnswer:",
        PromptStrategy.STRUCTURED_ANSWER: lambda query, _: f"Question: {query}\nAnswer (one word):",
        PromptStrategy.DIRECT_INSTRUCTION: lambda query, _: f"Please answer the following question in one word.\nQuestion: {query}\nAnswer:",
        PromptStrategy.CONTEXTUAL: lambda query, _: f"{query} (Please answer in one word)",
        PromptStrategy.QUESTION_ANSWER_PAIRS: lambda query, _: f"QSTN: What is the capital of France?\nANSR: Paris\nQSTN: What is the capital of Germany?\nANSR: Berlin\nQSTN: {query}\nANSR:",
        PromptStrategy.DIRECT_ANSWER: lambda query, _: f"Please provide a short, direct answer to the following question: {query} Answer:",
        PromptStrategy.QA_FORMAT: lambda query, _: f"Q: {query}\nA:",
        PromptStrategy.INSTRUCTIONAL: lambda query, _: f"Answer the following question in one or two words: {query}",
        PromptStrategy.SUMMARY: lambda query, _: f"Summarize the answer to the following question: {query}",
        PromptStrategy.ECHO: lambda query, _: f"{query} {query}",
        PromptStrategy.TRUE_COMPLETION: lambda query, _: f"{query} The true answer is:",
        PromptStrategy.DIRECT_COMPLETION: lambda query, _: f"{query} Answer:",
        PromptStrategy.ANSWER_COMPLETION: lambda query, _: f"{query} The correct answer is:",
        PromptStrategy.DIRECT_QUERY: lambda query, _: f"{query}?",
        PromptStrategy.FACTUAL_RETRIEVAL: lambda query, _: f"Based on known facts, what is the answer to the following: {query}?",
        PromptStrategy.FIRST_THOUGHT: lambda query, _: f"What is the first thing that comes to your mind when asked: {query}?",
        PromptStrategy.DEDUCTIVE_REASONING: lambda query, _: f"Given these facts: 1) Paris is the capital of France. 2) The Eiffel Tower is located in Paris. 3) French is the official language of France. Deduce the answer to the following: {query}.",
        PromptStrategy.EXPERT_PERSONA: get_expert_persona_prompt,
        PromptStrategy.REFLECTIVE_REASONING: get_reflective_reasoning_prompt,
        PromptStrategy.ZERO_SHOT: lambda query, _: f"Q: {query}\nA:",
        PromptStrategy.ONE_SHOT: lambda query, dataset_name: f"Q: {query}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][0][1]}\n\nQ:",
        PromptStrategy.TWO_SHOT: lambda query, dataset_name: f"Q: {query}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][0][1]}\n\nQ: {DATASET_EXAMPLE_QUERIES[dataset_name][1][0]}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][1][1]}\n\nQ:",
        PromptStrategy.THREE_SHOT: lambda query, dataset_name: f"Q: {query}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][0][1]}\n\nQ: {DATASET_EXAMPLE_QUERIES[dataset_name][1][0]}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][1][1]}\n\nQ: {DATASET_EXAMPLE_QUERIES[dataset_name][2][0]}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][2][1]}\n\nQ:",
        PromptStrategy.FOUR_SHOT: lambda query, dataset_name: f"Q: {query}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][0][1]}\n\nQ: {DATASET_EXAMPLE_QUERIES[dataset_name][1][0]}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][1][1]}\n\nQ: {DATASET_EXAMPLE_QUERIES[dataset_name][2][0]}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][2][1]}\n\nQ: {DATASET_EXAMPLE_QUERIES[dataset_name][3][0]}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][3][1]}\n\nQ:",
        PromptStrategy.FIVE_SHOT: lambda query, dataset_name: f"Q: {query}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][0][1]}\n\nQ: {DATASET_EXAMPLE_QUERIES[dataset_name][1][0]}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][1][1]}\n\nQ: {DATASET_EXAMPLE_QUERIES[dataset_name][2][0]}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][2][1]}\n\nQ: {DATASET_EXAMPLE_QUERIES[dataset_name][3][0]}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][3][1]}\n\nQ: {DATASET_EXAMPLE_QUERIES[dataset_name][4][0]}\nA: {DATASET_EXAMPLE_QUERIES[dataset_name][4][1]}\n\nQ:",
    }

    @classmethod
    def get_strategy(cls, strategy_type: PromptStrategy) -> Callable[[str, str], str]:
        """Get the prompt strategy function for given type."""
        if strategy_type not in cls.STRATEGIES:
            raise ValueError(f"Unknown prompt strategy: {strategy_type}")
        return cls.STRATEGIES[strategy_type]

    @classmethod
    def apply_strategy(cls, strategy_type: PromptStrategy, query: str, dataset_name: str) -> str:
        """Apply the specified prompt strategy to the query."""
        strategy_fn = cls.get_strategy(strategy_type)
        return strategy_fn(query, dataset_name)
