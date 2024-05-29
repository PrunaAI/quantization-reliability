from transformers import AutoModelForCausalLM, AutoTokenizer

def main(_run):
    model_name = _run.config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_text = "Once upon a time"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)

    generated_text = tokenizer.decode(outputs[0])
    print("Generated text:", generated_text)

if __name__ == "__main__":
    import seml
    from sacred import Experiment

    ex = Experiment("quantization_reliability")
    seml.setup_logger(ex)
    ex.main(main)
    ex.run_commandline()
