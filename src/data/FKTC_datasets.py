from src.data.ToyQADataset import toy_qa_dataset, format_toy_qa_dataset
from src.reliability.apply_typos import apply_typo_modifications

DATA_DIR = "/nfs/students/daro/data/MONITOR/FKTC"
DATA_FILES = [
    "P101",
    "P103",
    "P108",
    "P127",
    "P1376",
    "P1412",
    "P159",
    "P17",
    "P176",
    "P178",
    "P19",
    "P20",
    "P264",
    "P27",
    "P276",
    "P30",
    "P364",
    "P37",
    "P495",
    "P740"
]

DATA_FILES_JSON = [f"{file_name}-subclass.json" for file_name in DATA_FILES]

import os
import json
import random

def create_typo_dict(typo_type, intensity):
    base_dict = {
        "char_insertion": 0, "char_deletion": 0, "char_replacement": 0,
        "char_repetition": 0, "char_swapping": 0, "word_CMW": 0,
        "char_LCC": 0, "word_synonym": 0, "char_insert_noise": 0,
        "word_repeat": 0, "char_substitution": 0, "word_emoji": 0,
        "word_internet_slang": 0, "word_phrase_translation": 0,
        "word_context_aware_insertion": 0, "word_remove_punctuation": 0,
        "word_keyword_only": 0
    }
    
    if typo_type in base_dict:
        base_dict[typo_type] = intensity
    elif typo_type == "random":
        for _ in range(intensity):
            key = random.choice(list(base_dict.keys()))
            base_dict[key] += 1
    
    return base_dict

def load_question_answer_pairs(data_dir, dataset_names, max_relations=1, max_entries=None, taxonomy_type="0", typo_type="none", typo_intensity=0):
    question_answer_pairs = []
    
    for dataset_name in dataset_names.split(','):
        dataset_name = dataset_name.strip()
        file_path = os.path.join(data_dir, f"{dataset_name}-subclass.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf8') as f:
                lines = [json.loads(line) for line in f.readlines()]
                
                if len(lines) < 2:
                    print(f"Insufficient data in file: {file_path}")
                    continue
                
                relations = lines[0]['relations'][:max_relations]
                entries = lines[1:max_entries+1] if max_entries else lines[1:]
                
                for entry in entries:
                    subject = entry['subject']
                    answer = entry['object']
                    taxonomy = entry.get('taxonomy', [])
                    
                    for relation in relations:
                        question = construct_question(relation, subject, answer, taxonomy, taxonomy_type)
                        
                        if typo_type != "none":
                            typo_dict = create_typo_dict(typo_type, typo_intensity)
                            question = apply_typo_modifications(question, typo_dict)
                        
                        question_answer_pairs.append((question, answer))
        else:
            print(f"File not found: {file_path}")
    
    return question_answer_pairs

def construct_question(relation, subject, answer, taxonomy, taxonomy_type):
    if isinstance(taxonomy_type, int):
        taxonomy_type = str(taxonomy_type)
    if taxonomy_type == "0":
        return relation.replace("[X]", subject)
    elif taxonomy_type == "pos":
        return f"{answer}. {relation.replace('[X]', subject)}"
    else:
        index = int(taxonomy_type[3:]) - 1
        if index < len(taxonomy):
            fake_taxonomy = taxonomy[index]
        else:
            fake_taxonomy = taxonomy[-1]
        return f"{fake_taxonomy}. {relation.replace('[X]', subject)}"

def load_dataset_from_name(dataset_name, max_relations=1, max_entries=None, taxonomy_type="0", typo_type="none", typo_intensity=0):
    if dataset_name == "toy-qa-dataset":
        dataset = format_toy_qa_dataset(toy_qa_dataset, taxonomy_type=taxonomy_type)
        if typo_type != "none":
            typo_dict = create_typo_dict(typo_type, typo_intensity)
            dataset = [(apply_typo_modifications(q, typo_dict), a) for q, a in dataset]
        return dataset
    elif dataset_name in DATA_FILES:
        return load_question_answer_pairs(
            data_dir=DATA_DIR,
            dataset_names=dataset_name,
            max_relations=max_relations,
            max_entries=max_entries,
            taxonomy_type=taxonomy_type,
            typo_type=typo_type,
            typo_intensity=typo_intensity
        )
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")