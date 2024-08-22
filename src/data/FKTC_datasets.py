toy_qa_dataset = [
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

def load_question_answer_pairs(data_dir, dataset_names, max_relations=1, max_entries=None):
    """
    Load (question_string, answer_string) pairs from the corresponding -subclass.json files.

    Parameters:
        data_dir (str): Directory where the dataset files are stored.
        dataset_names (str): A comma-separated string of dataset names (e.g., 'P364', 'P37', 'P495').
        max_relations (int): The maximum number of relations to consider per entry. Default is 1.
        max_entries (int): The maximum number of entries to consider from each JSON file. Default is None (all entries).

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
                    
                    for relation in relations:
                        question = relation.replace("[X]", subject)
                        question_answer_pairs.append((question, answer))
        else:
            print(f"File not found: {file_path}")
    
    return question_answer_pairs

def load_dataset_from_name(dataset_name):
  if dataset_name == "toy_qa_dataset":
    return toy_qa_dataset
  elif dataset_name in DATA_FILES:
    return load_question_answer_pairs(DATA_DIR, dataset_name)
  else:
    raise ValueError(f"Invalid dataset name: {dataset_name}")