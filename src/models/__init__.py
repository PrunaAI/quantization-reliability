import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(model_name, weight_name=None, cache_dir=None, task=None, seed=123, directory_model=None, device="cuda"):
    """
    Load a model from the Huggingface transformers library.

    Parameters:
    - model_name (str): The name of the model to load.
    - weight_name (str): The specific weights to load for the model.
    - cache_dir (str): The directory to cache the model.
    - task (str): The task the model is to be used for.
    - seed (int): Random seed for reproducibility.
    - directory_model (str): The directory where the model is stored.
    - device (str): The device to load the model onto ('cuda' or 'cpu').

    Returns:
    - model: The loaded model.
    - tokenizer: The tokenizer associated with the model.
    """
    torch.manual_seed(seed)
    
    # Tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Model loading
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
    model.to(device)

    return model, tokenizer
