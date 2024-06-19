import os

def calculate_model_size(model_path):
    """
    This function calculates the total size of a model directory containing saved model files.

    Args:
        model_path (str): The path to the directory containing the model files.

    Returns:
        None: The function directly prints the total model size in human-readable format.
    """

    # Initialize total size variable
    total_size = 0

    # Loop through files in the model directory
    for filename in os.listdir(model_path):
        file_path = os.path.join(model_path, filename)
    # Check if it's a file (not a directory)
    if os.path.isfile(file_path):
        file_size = os.path.getsize(file_path)
        total_size += file_size

    # Convert to human-readable format
    if total_size > 1024**2:
        total_size_mb = total_size / (1024**2)
        print(f"Total Model Size for {model_path}: {total_size_mb:.2f} MB")
    elif total_size > 1024:
        total_size_kb = total_size / 1024
        print(f"Total Model Size for {model_path}: {total_size_kb:.2f} KB")
    else:
        print(f"Total Model Size for {model_path}: {total_size} bytes")  # Keep as bytes for small sizes

def prompt_large_language_model(model, tokenizer, input_text, device=None):
    """
    Prompts a large language model by generating text based on an input string.

    Args:
        model_name (str): Name of the pre-trained model from the Hugging Face Hub.
        input_text (str): Input string for the model to generate text from.
        device (str, optional): Device to use for computations (CPU or GPU). Defaults to "cuda" if available, otherwise "cpu".

    Returns:
        None: Prints the generated text to the console.
    """

    print(f"Prompt: {input_text}")
    try:
        # Convert input text to tensor and move to device
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        # Generate text using beam search (modify parameters as needed)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            max_length=512,  # Adjust maximum output length
            num_beams=5,  # Adjust number of beams for beam search
        )

        # Decode generated IDs back to text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Print generated text and elapsed time
        print(f"Generated text for model '{model.config._name_or_path}':\n{generated_text}")

    except Exception as e:
        print(f"An error occurred when generating text: {e}")