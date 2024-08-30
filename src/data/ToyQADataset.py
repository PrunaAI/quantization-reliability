toy_qa_dataset = [
    # Place / Region Names (3)
    {
        "query": "What is the location of the Great Wall of China?", 
        "answer": "China",
        "taxonomy": ["Mongolia", "Japan", "Vietnam", "Russia", "India"]
    },
    {
        "query": "Where is the Eiffel Tower located?", 
        "answer": "Paris",
        "taxonomy": ["Berlin", "Madrid", "Rome", "London", "Amsterdam"]
    },
    {
        "query": "What is the capital city of Japan?", 
        "answer": "Tokyo",
        "taxonomy": ["Kyoto", "Osaka", "Seoul", "Beijing", "Bangkok"]
    },
    
    # Adjectives (3)
    {
        "query": "How would you describe the taste of a lemon?", 
        "answer": "Sour",
        "taxonomy": ["Sweet", "Bitter", "Salty", "Umami", "Spicy"]
    },
    {
        "query": "What is the best way to describe the weather on a clear day?", 
        "answer": "Sunny",
        "taxonomy": ["Rainy", "Cloudy", "Stormy", "Windy", "Foggy"]
    },
    {
        "query": "How would you describe the feeling of touching velvet?", 
        "answer": "Soft",
        "taxonomy": ["Rough", "Sticky", "Slippery", "Hard", "Bumpy"]
    },
    
    # Person Names (3)
    {
        "query": "Who was the first president of the United States?", 
        "answer": "George Washington",
        "taxonomy": ["Thomas Jefferson", "Abraham Lincoln", "John Adams", "James Madison", "Alexander Hamilton"]
    },
    {
        "query": "Who is the author of 'Pride and Prejudice'?", 
        "answer": "Jane Austen",
        "taxonomy": ["Charlotte Brontë", "Mary Shelley", "Emily Dickinson", "George Eliot", "Louisa May Alcott"]
    },
    {
        "query": "Who painted the Mona Lisa?", 
        "answer": "Leonardo da Vinci",
        "taxonomy": ["Michelangelo", "Vincent van Gogh", "Claude Monet", "Pablo Picasso", "Raphael"]
    },
    
    # Abstract Answers (3)
    {
        "query": "What is your favorite color?", 
        "answer": "I don't know",
        "taxonomy": ["Blue", "Green", "Red", "Yellow", "Purple"]
    },
    {
        "query": "What do you think of the meaning of life?", 
        "answer": "It's complicated",
        "taxonomy": ["Happiness", "Love", "Success", "Knowledge", "Peace"]
    },
    {
        "query": "What comes after the end?", 
        "answer": "Nothing",
        "taxonomy": ["Infinity", "A new beginning", "The unknown", "Eternity", "Darkness"]
    },
    
    # Numerical Answers (3)
    {
        "query": "How many continents are there on Earth?", 
        "answer": "Seven",
        "taxonomy": ["Five", "Six", "Eight", "Nine", "Ten"]
    },
    {
        "query": "What is the boiling point of water in Celsius?", 
        "answer": "100",
        "taxonomy": ["0", "50", "90", "120", "150"]
    },
    {
        "query": "How many hours are there in a day?", 
        "answer": "24",
        "taxonomy": ["12", "48", "60", "72", "36"]
    },
    
    # Languages (3)
    {
        "query": "What language is primarily spoken in Brazil?", 
        "answer": "Portuguese",
        "taxonomy": ["Spanish", "French", "English", "Italian", "Dutch"]
    },
    {
        "query": "Which language is spoken in Germany?", 
        "answer": "German",
        "taxonomy": ["Dutch", "French", "English", "Danish", "Swedish"]
    },
    {
        "query": "What is the official language of China?", 
        "answer": "Mandarin",
        "taxonomy": ["Cantonese", "Japanese", "Korean", "Thai", "Hindi"]
    },
    
    # Company Names (3)
    {
        "query": "Which company developed the iPhone?", 
        "answer": "Apple",
        "taxonomy": ["Samsung", "Google", "Microsoft", "Huawei", "Sony"]
    },
    {
        "query": "What is the name of the online retailer founded by Jeff Bezos?", 
        "answer": "Amazon",
        "taxonomy": ["eBay", "Alibaba", "Walmart", "Flipkart", "Target"]
    },
    {
        "query": "Which company is known for its search engine?", 
        "answer": "Google",
        "taxonomy": ["Yahoo", "Bing", "DuckDuckGo", "Baidu", "Ask"]
    }
]

def format_toy_qa_dataset(toy_qa_dataset, taxonomy_type='0'):
    formatted_dataset = []

    # Mapping taxonomy types to their indices
    taxonomy_map = {
        '0': None,        # No taxonomy
        'pos': 'answer',  # Use the real answer
        'neg1': 0,        # Use first taxonomy distractor
        'neg2': 1,        # Use second taxonomy distractor
        'neg3': 2,        # Use third taxonomy distractor
        'neg4': 3,        # Use fourth taxonomy distractor
        'neg5': 4         # Use fifth taxonomy distractor
    }

    for entry in toy_qa_dataset:
        query = entry['query']
        answer = entry['answer']
        taxonomy = entry['taxonomy']

        # Determine which taxonomy to use
        if taxonomy_type == '0':
            # No taxonomy applied
            formatted_query = query
        elif taxonomy_type == 'pos':
            # Use the real answer as the taxonomy
            formatted_query = f"{answer}. {query}"
        else:
            # Use the corresponding taxonomy distractor
            taxonomy_index = taxonomy_map[taxonomy_type]
            selected_taxonomy = taxonomy[taxonomy_index]
            formatted_query = f"{selected_taxonomy}. {query}"

        formatted_dataset.append((formatted_query, answer))

    return formatted_dataset