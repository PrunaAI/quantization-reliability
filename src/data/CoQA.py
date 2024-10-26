import json
from typing import List, Tuple, Dict, Optional
# Import the typo modification function from the existing codebase
from src.data.constants import COQA_PATH
from src.reliability.apply_typos import apply_typo_modifications
from src.reliability.create_typos_list import create_typo_dict


def load_coqa_dataset(
    file_path: str = COQA_PATH,
    max_entries: Optional[int] = None,
    typo_type: str = "none",
    typo_intensity: int = 0,
    concatenate_qa: bool = False
) -> List[Tuple[str, str, str]]:
    """
    Loads the CoQA dataset and returns a list of (story, question, answer) tuples.
    Processes the data in the same way as the semantic uncertainty script.
    
    Args:
        file_path: Path to the CoQA JSON file
        max_entries: Maximum number of QA pairs to load (None for all)
        typo_type: Type of typo to apply ("none" for no typos)
        typo_intensity: Intensity of typo modifications
        concatenate_qa: If True, concatenates previous Q&A pairs into the question string
        
    Returns:
        List of (story, question, answer) tuples
    """
    qa_triplets = []
    
    # Load the CoQA dataset
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    
    # Create typo dictionary if needed
    typo_dict = None
    if typo_type != "none":
        typo_dict = create_typo_dict(typo_type, typo_intensity)
    
    # Process each story in the dataset
    for sample in data:
        if max_entries and len(qa_triplets) >= max_entries:
            break
            
        story = sample['story']
        questions = sample['questions']
        answers = sample['answers']
        
        if concatenate_qa:
            # Initialize the concatenated question history for this story
            qa_history = ""
            
            # Process each question for the current story
            for question_index, (question, answer) in enumerate(zip(questions, answers)):
                question_text = question['input_text']
                answer_text = answer['input_text']
                
                if typo_dict is not None:
                    question_text = apply_typo_modifications(question_text, typo_dict, [answer_text])
                
                # For the first question, just add the story and question
                if question_index == 0:
                    current_question = f"{story}. Q: {question_text}"
                else:
                    # Add the previous Q&A pair to the history and create new question
                    qa_history += f"Q: {questions[question_index-1]['input_text']}"
                    qa_history += f", A: {answers[question_index-1]['input_text']}, "
                    current_question = f"{story}. {qa_history}Q: {question_text}"
                
                qa_triplets.append((
                    story,  # Original story
                    current_question,  # Concatenated history + current question
                    answer_text  # Current answer
                ))
                
                if max_entries and len(qa_triplets) >= max_entries:
                    break
                    
        else:
            # Original behavior: process each Q&A pair separately
            current_context = story
            
            for question_index, (question, answer) in enumerate(zip(questions, answers)):
                question_text = question['input_text']
                answer_text = answer['input_text']
                
                if typo_dict is not None:
                    question_text = apply_typo_modifications(question_text, typo_dict, [answer_text])
                
                qa_triplets.append((
                    current_context,
                    question_text,
                    answer_text
                ))
                
                # Update the context with this Q&A pair for the next question
                if not current_context.endswith('.'):
                    current_context += '.'
                current_context += f" Q: {question_text} A: {answer_text}"
                
                if max_entries and len(qa_triplets) >= max_entries:
                    break
    
    return qa_triplets

def load_coqa_dataset_pairs(
    file_path: str = COQA_PATH,
    max_entries: Optional[int] = None,
    typo_type: str = "none",
    typo_intensity: int = 0
) -> List[Tuple[str, str]]:
    """
    Loads the CoQA dataset and returns a list of (question, answer) pairs,
    with previous Q&A pairs concatenated into the question string.
    
    Args:
        file_path: Path to the CoQA JSON file
        max_entries: Maximum number of QA pairs to load (None for all)
        typo_type: Type of typo to apply ("none" for no typos)
        typo_intensity: Intensity of typo modifications
        
    Returns:
        List of (question, answer) tuples with concatenated history
    """
    qa_triplets = load_coqa_dataset(
        file_path=file_path,
        max_entries=max_entries,
        typo_type=typo_type,
        typo_intensity=typo_intensity,
        concatenate_qa=True
    )
    
    return [(question, answer) for _, question, answer in qa_triplets]