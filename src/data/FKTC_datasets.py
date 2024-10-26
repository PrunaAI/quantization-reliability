from src.data.ToyQADataset import toy_qa_dataset
from src.data.constants import DATA_DIR, DATA_FILES
from src.reliability.apply_typos import apply_typo_modifications

import os
import json
import random

from src.reliability.create_typos_list import create_typo_dict


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
                        question = relation.replace("[X]", subject)
                        
                        if typo_type != "none":
                            typo_dict = create_typo_dict(typo_type, typo_intensity)
                            question = apply_typo_modifications(question, typo_dict, taxonomy + [answer])
                        
                        question_answer_pairs.append((question, answer))
        else:
            print(f"File not found: {file_path}")
    
    return question_answer_pairs

def load_fktc(dataset_name, max_relations=1, max_entries=None, typo_type="none", typo_intensity=0):
    if dataset_name == "toy-qa-dataset":
        dataset = toy_qa_dataset
        if typo_type != "none":
            typo_dict = create_typo_dict(typo_type, typo_intensity)
            dataset = [(apply_typo_modifications(q, typo_dict, [a]), a) for q, a in dataset]
        return dataset
    elif dataset_name in DATA_FILES:
        return load_question_answer_pairs(
            data_dir=DATA_DIR,
            dataset_names=dataset_name,
            max_relations=max_relations,
            max_entries=max_entries,
            typo_type=typo_type,
            typo_intensity=typo_intensity
        )
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")