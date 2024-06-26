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