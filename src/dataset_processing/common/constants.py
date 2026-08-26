from src.dataset_processing.perturbations.enums import PerturbationType


ALL_PERTURBATION_INTENSITIES={
    # PerturbationType.WORD_SYNONYM: [1],
    PerturbationType.WORD_INSERTION: [1, 2, 3],
    PerturbationType.WORD_DELETION: [1, 2, 3],
    PerturbationType.WORD_SWAPPING: [1, 2, 3],
    PerturbationType.WORD_REPETITION: [1, 2, 3],
    PerturbationType.WORD_INTERNET_SLANG: [1, 2, 3],
    PerturbationType.WORD_PHRASE_TRANSLATION: [1, 2, 3],
    # PerturbationType.WORD_TAXONOMY_POS: [1, 2, 3],
    # PerturbationType.WORD_TAXONOMY_NEG: [1, 2, 3],
    PerturbationType.CHAR_INSERTION: [1, 2, 3],
    PerturbationType.CHAR_DELETION: [1, 2, 3],
    PerturbationType.CHAR_REPLACEMENT: [1, 2, 3],
    PerturbationType.CHAR_SWAPPING: [1, 2, 3],
    PerturbationType.CHAR_REPETITION: [1, 2, 3],
    PerturbationType.CHAR_SUBSTITUTION: [1, 2, 3],
    PerturbationType.CHAR_INSERT_NOISE: [1, 2, 3],
    PerturbationType.CHAR_CASE_CHANGE: [1, 2, 3],
    PerturbationType.CHAR_EMOJI: [1, 2, 3]
}

FKTC_DATASET_FILES = [
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

DATA_FILES = ["fktc"] + FKTC_DATASET_FILES

SEVEN_SHOT_EXAMPLES = [
    {
        "question": "What do people use to absorb extra ink from a fountain pen?",
        "choices": {"text": ["shirt pocket", "calligrapher's hand", "inkwell", "desk drawer", "blotter"]},
        "answer_key": "E",
        "explanation": "The answer must be an item that can absorb ink. Of the above choices, only blotters are used to absorb ink."
    },
    {
        "question": "What home entertainment equipment requires cable?",
        "choices": {"text": ["radio shack", "substation", "television", "cabinet"]},
        "answer_key": "C",
        "explanation": "The answer must require cable. Of the above choices, only television requires cable."
    },
    {
        "question": "The fox walked from the city into the forest, what was it looking for?",
        "choices": {"text": ["pretty flowers", "hen house", "natural habitat", "storybook"]},
        "answer_key": "C",
        "explanation": "The answer must be something in the forest. Of the above choices, only natural habitat is in the forest."
    },
    {
        "question": "Sammy wanted to go to where the people were. Where might he go?",
        "choices": {"text": ["populated areas", "race track", "desert", "apartment", "roadblock"]},
        "answer_key": "A",
        "explanation": "The answer must be a place with a lot of people. Of the above choices, only populated areas have a lot of people."
    },
    {
        "question": "Where do you put your grapes just before checking out?",
        "choices": {"text": ["mouth", "grocery cart", "super market", "fruit basket", "fruit market"]},
        "answer_key": "B",
        "explanation": "The answer should be the place where grocery items are placed before checking out. Of the above choices, grocery cart makes the most sense for holding grocery items."
    },
    {
        "question": "Google Maps and other highway and street GPS services have replaced what?",
        "choices": {"text": ["united states", "mexico", "countryside", "atlas"]},
        "answer_key": "D",
        "explanation": "The answer must be something that used to do what Google Maps and GPS services do, which is to give directions. Of the above choices, only atlases are used to give directions."
    },
    {
        "question": "Before getting a divorce, what did the wife feel who was doing all the work?",
        "choices": {"text": ["harder", "anguish", "bitterness", "tears", "sadness"]},
        "answer_key": "C",
        "explanation": "The answer should be the feeling of someone getting divorced who was doing all the work. Of the above choices, the closest feeling is bitterness."
    }
]
