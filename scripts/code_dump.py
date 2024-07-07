import torch

dataset = [{"text": text} for text in wikitext_dataset]

samples = []
n_run = 0
n_samples=512
awq_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

for data in dataset:
    if isinstance(data, list):
        line_encoded = data
    else:
        line = data["text"]
        line = line.strip()
        line_encoded = awq_tokenizer.encode(line)
    if len(line_encoded) > 512:
        continue
    sample = torch.tensor([line_encoded])
    if sample.numel() == 0:
        continue
    samples.append(sample)
    n_run += 1
    if n_run == n_samples:
        break
# now concatenate all samples and split according to block size
print(samples)
cat_samples = torch.cat(samples, dim=1)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, AwqConfig
from accelerate.utils import load_and_quantize_model
from accelerate import init_empty_weights

device="cuda"

# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # For instruction-based models.
# model_name = "meta-llama/Meta-Llama-3-8B"  # Too large to run on a gpu_gtx1080. GPU gpu_a100 is required.
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Small enough to run on a gpu_gtx1080.
# model_name = "microsoft/Phi-3-vision-128k-instruct"  # Small enough to run on a gpu_gtx1080.

tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    
# Setting quantization bits for the model
weight_quantization_bits = 8
double_quant = False

bnb_config = BitsAndBytesConfig(
    load_in_8bit=(weight_quantization_bits == 8),
    load_in_4bit=(weight_quantization_bits == 4),
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_head"],
    llm_int8_enable_fp32_cpu_offload=False,
    llm_int8_has_fp16_weight=False,
    # bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="fp4",
    # bnb_4bit_quant_type="nf4"
    bnb_4bit_use_double_quant=double_quant,
)

bnb_config.skip_modules = None

smashed_model_bnb = load_and_quantize_model(model, bnb_quantization_config=bnb_config, device_map=device)
print(smashed_model_bnb.__class__.__name__)

# Calibration Dataset needed - WikiText? Something else because data leakage? TODO: Explore
# smashed_model_awq = AutoModelForCausalLM.from_pretrained(
#     temp_dir, quantization_config=awq_config, trust_remote_code=True
# )

awq_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=512,
    do_fuse=False,
)

model.context_length()

import accelerate
print(accelerate.__file__)

import transformers
print(transformers.__file__)

from transformers import BitsAndBytesConfig
print(BitsAndBytesConfig.__dict__)

#!pip index versions accelerate
#!pip install accelerate --force-reinstall
!pip show transformers
!pip index versions transformers
!pip show accelerate
!pip index versions accelerate

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)

import numpy as np
import evaluate

brier_score = evaluate.load("brier_score")
predictions = np.array([0, 0, 1, 1])
references = np.array([0.1, 0.9, 0.8, 0.3])
results = brier_score.compute(predictions=predictions, references=references)
print(results)

def brier_score(Y, alpha):
    batch_size = alpha.size(0)

    p = torch.nn.functional.normalize(alpha, p=1, dim=-1)
    indices = torch.arange(batch_size)
    p[indices, Y.squeeze()] -= 1
    brier_score = p.norm(dim=-1).mean().cpu().detach().numpy()
    return brier_score

import torch
import tqdm

def evaluate_perplexity(model, tokenizer, dataloader, device="cuda"):
    model.eval()
    
    nlls = []
    for batch in tqdm.tqdm(dataloader):
        input_ids, target_ids = batch
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

    # print(len(nlls))
    # print(nlls)
    # print(torch.stack(nlls))
    # print(torch.stack(nlls).mean())
    # print(torch.exp(torch.stack(nlls).mean()))
    ppl = torch.exp(torch.stack(nlls).mean())
    print(f"Perplexity of model {model.NAME}: {ppl:.2f}")
    
    return ppl

evaluate_perplexity(model, tokenizer, wikitext_dataloader, device="cuda")

import numpy as np

logits = outputs["logits"][0].cpu().numpy()
sequences = logits.reshape(-1, 512, logits.shape[-1])
argmax_indices = np.argmax(sequences, axis=2)

print(tokenizer.decode(labels[0][0:25], skip_special_tokens=True))
print(tokenizer.decode(labels[0][0:25], skip_special_tokens=True))
print(tokenizer.decode(argmax_indices[0][0:25], skip_special_tokens=True))

        # print(f"Step {begin_loc}: NLL = {neg_log_likelihood:.2f}")
        # import numpy as np

        # logits = outputs["logits"][0].cpu().numpy()
        # sequences = logits.reshape(-1, 2048, logits.shape[-1])
        # argmax_indices = np.argmax(sequences, axis=2)

        # print(f"Input: {tokenizer.decode(input_ids[0][-20:], skip_special_tokens=True)}")
        # print(f"Target: {tokenizer.decode(target_ids[0][-20:], skip_special_tokens=True)}")
        # print(f"Prediction: {tokenizer.decode(argmax_indices[0][-20:], skip_special_tokens=True)}")

print(wikitext_data_module.val_dataset["text"][:100])
print(wikitext_dataloader.dataset.dataset["text"][:100])

print(len(wikitext_data_module.val_dataset["text"]))
print(len(wikitext_dataloader.dataset.dataset["text"]))

wikitext_dataset = []

# Loop through each batch in the dataloader
for batch in wikitext_dataloader:
  # Assuming the batch contains input_ids (tokenized text) and labels
  input_ids, labels = batch
  
  # Decode the input IDs back to text using the tokenizer
  decoded_text = wikitext_data_module.tokenizer.decode(input_ids[0].tolist())  # Assuming first element in batch
  wikitext_dataset.append(decoded_text)
  
print(f"Sample texts from train dataloader: {wikitext_dataset[1][:1000]}")
print(f"Type: {type(wikitext_dataset)}, Length: {len(wikitext_dataset)}")

oasst_dataset = []

# Loop through each batch in the dataloader
for batch in oasst_dataloader:
  # Assuming the batch contains input_ids (tokenized text) and labels
  input_ids, labels = batch
  
  # Decode the input IDs back to text using the tokenizer
  decoded_text = oasst_data_module.tokenizer.decode(input_ids[0].tolist())  # Assuming first element in batch
  oasst_dataset.append(decoded_text)
  
print(f"Sample texts from train dataloader: {oasst_dataset[1][:1000]}")
print(f"Type: {type(oasst_dataset)}, Length: {len(oasst_dataset)}")

%reset -f
import gc
gc.collect()

import torch
import torchmetrics
import tqdm
from torch.cuda.amp import autocast, GradScaler

with torch.no_grad():
    torch.cuda.empty_cache()

def evaluate_perplexity_v2(model, dataloader, stride=512, max_length=None, device="cuda", to_device=False):
    if isinstance(model, torch.nn.Module):
        model.eval()
        print(f"Model in evaluation mode. Device: {device}")
    if to_device:
        model.to(device)

    metric = torchmetrics.text.Perplexity(ignore_index=-100).to(device)  # -100 is the padding token.
    !nvidia-smi --query-gpu=memory.free --format=csv | tail -n +2 | awk -F ' ' '{print "Free GPU Memory (GB):", $1 / 1024}'
    
    for i, (x, y) in enumerate(dataloader):
        print(f"Processing batch {i}")
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad() and autocast():
            outputs = model(x)
            logits = outputs.logits
            !nvidia-smi --query-gpu=memory.free --format=csv | tail -n +2 | awk -F ' ' '{print "Free GPU Memory (GB):", $1 / 1024}'
            
            # Metric on current batch
            perplexity = metric(logits.float(), y)
            print(f"Perplexity: {perplexity:.2f}")
            
        del outputs, logits, perplexity
        torch.cuda.empty_cache()

    # Metric on all batches using custom accumulation
    perplexity = metric.compute()
    return perplexity.item()

wikitext_dataloader = wikitext_data_module.val_dataloader()
perpl = evaluate_perplexity_v2(model=model, dataloader=wikitext_dataloader, device="cuda")
print(f"Perplexity: {perpl:.2f}")

print(input_ids_list[0].shape)
print(len(input_ids_list))
print(len(wikitext_dataloader))

print(tokenizer.decode(input_ids_list[0][0][:50], skip_special_tokens=True))
# print(encodings.input_ids[:, :50])
print(tokenizer.decode(encodings.input_ids[:, :50][0], skip_special_tokens=True))

for i, (x, y) in enumerate(wikitext_dataloader):
    if i < 1:
        print(tokenizer.decode(x[0][:50], skip_special_tokens=True))
        print(tokenizer.decode(y[0][:50], skip_special_tokens=True))
        
from tqdm import tqdm

from datasets import load_dataset
test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
encodings = tokenizer('\n\n'.join(test['text']), return_tensors='pt')

max_length = tokenizer.model_max_length
stride = 1024

lls = []
input_ids_list = []
target_ids_list = []
for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
    # if i > stride:
    #     break
    begin_loc = max(i + stride - max_length, 0)
    end_loc = i + stride
    input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:,:-stride] = -100
    input_ids_list.append(input_ids)
    target_ids_list.append(target_ids)
    
import logging
from torch.cuda.amp import autocast
import torch.nn.functional as F
import torch

def evaluate_brier_score(model, dataloader, device="cuda", to_device=False):
    if to_device:
        model.to(device)
    if isinstance(model, torch.nn.Module):
        model.eval()
        logging.info(f"Model in evaluation mode. Device: {device}")
        
    brier_sum = 0

    for i, (x, y) in enumerate(dataloader):
        if i > 10:
            break
        print(f"Processing batch {i}")
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad() and autocast():
            outputs = model(x)
            logits = outputs.logits
            !nvidia-smi --query-gpu=memory.free --format=csv | tail -n +2 | awk -F ' ' '{print "Free GPU Memory (GB):", $1 / 1024}'
            
            # Shift logits and target_ids to the left by 1 for calculating the Brier score
            shifted_logits = logits[:, :-1].contiguous()
            shifted_target_ids = x[:, 1:].contiguous()

            # Flatten the logits and target_ids for calculation
            shifted_logits = shifted_logits.view(-1, shifted_logits.size(-1))
            shifted_target_ids = shifted_target_ids.view(-1)

            # Filter out the -100 targets
            valid_indices = shifted_target_ids != -100
            valid_logits = shifted_logits[valid_indices]
            valid_target_ids = shifted_target_ids[valid_indices]

            # Get the probabilities
            probs = F.softmax(valid_logits, dim=-1)

            # Create one-hot target vectors
            targets = F.one_hot(valid_target_ids, num_classes=probs.size(-1)).float()

            # Calculate the Brier score
            brier_score = torch.mean((probs - targets) ** 2)
            print(f"Brier Score: {brier_score:.4f}")
            brier_sum += brier_score
            del brier_score

    avg_brier_score = brier_sum / len(dataloader)
    print(f"Final Brier Score: {avg_brier_score:.4f}")
    
    return avg_brier_score

wikitext_dataloader = wikitext_data_module.test_dataloader()
evaluate_brier_score(model, wikitext_dataloader, device=device)