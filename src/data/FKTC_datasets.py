from src.data.ToyQADataset import toy_qa_dataset, format_toy_qa_dataset

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

def load_question_answer_pairs(data_dir, dataset_names, max_relations=1, max_entries=None, taxonomy_type="0"):
    """
    Load (question_string, answer_string) pairs from the corresponding -subclass.json files.

    Parameters:
        data_dir (str): Directory where the dataset files are stored.
        dataset_names (str): A comma-separated string of dataset names (e.g., 'P364', 'P37', 'P495').
        max_relations (int): The maximum number of relations to consider per entry. Default is 1.
        max_entries (int): The maximum number of entries to consider from each JSON file. Default is None (all entries).
        taxonomy_type (str): The type of taxonomy to apply: "0", "pos", "neg1", "neg2", etc. Default is "0".

    Returns:
        list: A list of tuples where each tuple contains a question string and an answer string.
    """
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
                
                relations = lines[0]['relations'][:max_relations]  # Limit the number of relations
                entries = lines[1:max_entries+1] if max_entries else lines[1:]  # Limit the number of entries
                
                for entry in entries:
                    subject = entry['subject']
                    answer = entry['object']
                    taxonomy = entry.get('taxonomy', [])
                    
                    for relation in relations:
                        if taxonomy_type == "0":
                            # No taxonomy applied
                            question = relation.replace("[X]", subject)
                        elif taxonomy_type == "pos":
                            # Positive taxonomy: prepend the answer
                            question = f"{answer}. {relation.replace('[X]', subject)}"
                        else:
                            # Negative taxonomy: extract index from taxonomy_type ("neg1" -> 0, "neg2" -> 1, etc.)
                            index = int(taxonomy_type[3:]) - 1
                            if index < len(taxonomy):
                                fake_taxonomy = taxonomy[index]
                                question = f"{fake_taxonomy}. {relation.replace('[X]', subject)}"
                            else:
                                # Skip if the taxonomy index is out of bounds
                                continue
                        
                        question_answer_pairs.append((question, answer))
        else:
            print(f"File not found: {file_path}")
    
    return question_answer_pairs

def load_dataset_from_name(dataset_name, max_relations=1, max_entries=None, taxonomy_type="0"):
    if dataset_name == "toy-qa-dataset":
        return format_toy_qa_dataset(toy_qa_dataset, taxonomy_type=taxonomy_type)
    elif dataset_name in DATA_FILES:
        return load_question_answer_pairs(
            data_dir=DATA_DIR,
            dataset_names=dataset_name,
            max_relations=max_relations,
            max_entries=max_entries,
            taxonomy_type=taxonomy_type
        )
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")