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

# Old TextDataset
class TextDataset(Dataset):
    def __init__(self, dataset, tokenizer, n_samples=None, sequence_length=2048, stride=512):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.texts = dataset["text"]
        if n_samples is not None:
            self.texts = self.texts[:n_samples]
        tokenized_dataset = self.tokenizer("\n\n".join(self.texts), return_tensors="pt")
        self.data = tokenized_dataset.input_ids[0, :-1]
        self.labels = tokenized_dataset.input_ids[0]
        self.sequence_length = sequence_length
        self.stride = stride

    def __len__(self):
        return len(self.data) // self.sequence_length

    def __getitem__(self, index):
        start_index = max(index * self.stride + self.stride - self.sequence_length, 0)
        end_index = start_index + self.stride
        if end_index > len(self.data):
            raise IndexError("Index out of bounds")
        input_ids = self.data[start_index:end_index]
        target_ids = self.labels[start_index + 1: end_index + 1]
        target_ids[:-self.stride] = -100
        
        return input_ids, target_ids
    
emoji_dict = {
    "smile": "😊",
    "smiles": "😊",
    "smiling": "😊",
    "laugh": "😂",
    "laughs": "😂",
    "laughing": "😂",
    "sad": "😢",
    "sadness": "😢",
    "crying": "😢",
    "angry": "😠",
    "angered": "😠",
    "anger": "😠",
    "love": "❤️",
    "loves": "❤️",
    "loving": "❤️",
    "heart": "❤️",
    "hearts": "❤️",
    "sun": "☀️",
    "sunny": "☀️",
    "sunshine": "☀️",
    "moon": "🌙",
    "moonlight": "🌙",
    "star": "⭐",
    "stars": "⭐",
    "starry": "⭐",
    "rain": "🌧️",
    "rains": "🌧️",
    "rainy": "🌧️",
    "cloud": "☁️",
    "clouds": "☁️",
    "cloudy": "☁️",
    "snow": "❄️",
    "snowy": "❄️",
    "snowing": "❄️",
    "fire": "🔥",
    "fiery": "🔥",
    "hot": "🔥",
    "cold": "🥶",
    "freezing": "🥶",
    "ice": "🧊",
    "icy": "🧊",
    "tree": "🌳",
    "trees": "🌳",
    "forest": "🌳",
    "flower": "🌸",
    "flowers": "🌸",
    "flowering": "🌸",
    "dog": "🐶",
    "dogs": "🐶",
    "cat": "🐱",
    "cats": "🐱",
    "bird": "🐦",
    "birds": "🐦",
    "fish": "🐠",
    "fishes": "🐠",
    "car": "🚗",
    "cars": "🚗",
    "bus": "🚌",
    "buses": "🚌",
    "train": "🚂",
    "trains": "🚂",
    "airplane": "✈️",
    "airplanes": "✈️",
    "flying": "✈️",
    "boat": "🚢",
    "boats": "🚢",
    "sailing": "🚢",
    "house": "🏠",
    "houses": "🏠",
    "building": "🏢",
    "buildings": "🏢",
    "food": "🍔",
    "eat": "🍔",
    "eating": "🍔",
    "drink": "🥤",
    "drinks": "🥤",
    "drinking": "🥤",
    "music": "🎵",
    "musical": "🎵",
    "song": "🎵",
    "book": "📚",
    "books": "📚",
    "reading": "📚",
    "movie": "🎬",
    "movies": "🎬",
    "film": "🎬",
    "phone": "📱",
    "phones": "📱",
    "calling": "📱",
    "computer": "💻",
    "computers": "💻",
    "computing": "💻",
    "money": "💰",
    "rich": "💰",
    "wealth": "💰",
    "clock": "🕰️",
    "clocks": "🕰️",
    "time": "🕰️",
    "gift": "🎁",
    "gifts": "🎁",
    "present": "🎁",
    "presents": "🎁",
    "birthday": "🎂",
    "birthdays": "🎂",
    "celebrate": "🎉",
    "celebrating": "🎉",
    "celebration": "🎉",
    "win": "🏆",
    "winning": "🏆",
    "winner": "🏆",
    "lose": "😞",
    "losing": "😞",
    "loser": "😞",
    "sport": "⚽",
    "sports": "⚽",
    "athletic": "⚽",
    "sleep": "😴",
    "sleeping": "😴",
    "sleepy": "😴",
    "work": "💼",
    "working": "💼",
    "job": "💼",
    "school": "🏫",
    "studying": "🏫",
    "learn": "🏫",
    "vacation": "🏖️",
    "vacationing": "🏖️",
    "holiday": "🏖️",
    "travel": "✈️",
    "traveling": "✈️",
    "journey": "✈️",
    "idea": "💡",
    "ideas": "💡",
    "thinking": "💡",
    "question": "❓",
    "questions": "❓",
    "questioning": "❓",
    "answer": "✅",
    "answers": "✅",
    "answering": "✅",
    "warning": "⚠️",
    "warnings": "⚠️",
    "caution": "⚠️",
    "stop": "🛑",
    "stopping": "🛑",
    "halt": "🛑",
    "go": "✅",
    "going": "✅",
    "start": "✅",
    "hello": "👋",
    "hi": "👋",
    "greeting": "👋",
    "goodbye": "👋",
    "bye": "👋",
    "farewell": "👋",
    "yes": "👍",
    "agree": "👍",
    "agreeing": "👍",
    "no": "👎",
    "disagree": "👎",
    "disagreeing": "👎",
    "ok": "👌",
    "okay": "👌",
    "fine": "👌",
    "good": "😊",
    "great": "😊",
    "excellent": "😊",
    "bad": "😞",
    "terrible": "😞",
    "awful": "😞",
    "pizza": "🍕",
    "pizzas": "🍕",
    "hamburger": "🍔",
    "hamburgers": "🍔",
    "burger": "🍔",
    "burgers": "🍔",
    "fries": "🍟",
    "french fries": "🍟",
    "sushi": "🍣",
    "rice": "🍚",
    "noodles": "🍜",
    "ramen": "🍜",
    "taco": "🌮",
    "tacos": "🌮",
    "burrito": "🌯",
    "burritos": "🌯",
    "egg": "🥚",
    "eggs": "🥚",
    "bread": "🍞",
    "sandwich": "🥪",
    "sandwiches": "🥪",
    "cake": "🎂",
    "cakes": "🎂",
    "cookie": "🍪",
    "cookies": "🍪",
    "candy": "🍬",
    "candies": "🍬",
    "lollipop": "🍭",
    "lollipops": "🍭",
    "ice cream": "🍦",
    "coffee": "☕",
    "tea": "🍵",
    "milk": "🥛",
    "beer": "🍺",
    "beers": "🍺",
    "wine": "🍷",
    "cocktail": "🍸",
    "cocktails": "🍸",
    "fruit": "🍎",
    "fruits": "🍎",
    "apple": "🍎",
    "apples": "🍎",
    "banana": "🍌",
    "bananas": "🍌",
    "orange": "🍊",
    "oranges": "🍊",
    "lemon": "🍋",
    "lemons": "🍋",
    "strawberry": "🍓",
    "strawberries": "🍓",
    "watermelon": "🍉",
    "watermelons": "🍉",
    "grapes": "🍇",
    "pineapple": "🍍",
    "pineapples": "🍍",
    "peach": "🍑",
    "peaches": "🍑",
    "cherry": "🍒",
    "cherries": "🍒",
    "vegetable": "🥕",
    "vegetables": "🥕",
    "carrot": "🥕",
    "carrots": "🥕",
    "broccoli": "🥦",
    "tomato": "🍅",
    "tomatoes": "🍅",
    "potato": "🥔",
    "potatoes": "🥔",
    "corn": "🌽",
    "mushroom": "🍄",
    "mushrooms": "🍄",
    "avocado": "🥑",
    "avocados": "🥑",
    "eggplant": "🍆",
    "eggplants": "🍆",
    "cucumber": "🥒",
    "cucumbers": "🥒",
    "pepper": "🌶️",
    "peppers": "🌶️",
    "garlic": "🧄",
    "onion": "🧅",
    "onions": "🧅",
    "peanut": "🥜",
    "peanuts": "🥜",
    "chestnut": "🌰",
    "chestnuts": "🌰",
    "bacon": "🥓",
    "salad": "🥗",
    "salads": "🥗",
    "popcorn": "🍿",
    "butter": "🧈",
    "salt": "🧂",
    "salty": "🧂",
    "brain": "🧠",
    "brains": "🧠",
    "bone": "🦴",
    "bones": "🦴",
    "eyes": "👀",
    "eye": "👁️",
    "ear": "👂",
    "ears": "👂",
    "nose": "👃",
    "noses": "👃",
    "mouth": "👄",
    "mouths": "👄",
    "tongue": "👅",
    "tongues": "👅",
    "baby": "👶",
    "babies": "👶",
    "child": "🧒",
    "children": "🧒",
    "boy": "👦",
    "boys": "👦",
    "girl": "👧",
    "girls": "👧",
    "man": "👨",
    "men": "👨",
    "woman": "👩",
    "women": "👩",
    "person": "🧑",
    "people": "🧑",
    "family": "👪",
    "families": "👪",
    "couple": "👫",
    "couples": "👫",
    "bride": "👰",
    "groom": "🤵",
    "grandma": "👵",
    "grandmother": "👵",
    "grandpa": "👴",
    "grandfather": "👴",
    "princess": "👸",
    "prince": "🤴",
    "santa": "🎅",
    "ghost": "👻",
    "ghosts": "👻",
    "alien": "👽",
    "aliens": "👽",
    "robot": "🤖",
    "robots": "🤖",
    "zombie": "🧟",
    "zombies": "🧟",
    "footprint": "👣",
    "footprints": "👣",
    "monkey": "🐵",
    "monkeys": "🐵",
    "gorilla": "🦍",
    "gorillas": "🦍",
    "orangutan": "🦧",
    "orangutans": "🦧",
    "panda": "🐼",
    "pandas": "🐼",
    "sloth": "🦥",
    "sloths": "🦥",
    "otter": "🦦",
    "otters": "🦦",
    "skunk": "🦨",
    "skunks": "🦨",
    "kangaroo": "🦘",
    "kangaroos": "🦘",
    "badger": "🦡",
    "badgers": "🦡",
    "paw": "🐾",
    "paws": "🐾",
    "turkey": "🦃",
    "turkeys": "🦃",
    "chicken": "🐔",
    "chickens": "🐔",
    "rooster": "🐓",
    "roosters": "🐓",
    "penguin": "🐧",
    "penguins": "🐧",
    "dove": "🕊️",
    "doves": "🕊️",
    "eagle": "🦅",
    "eagles": "🦅",
    "duck": "🦆",
    "ducks": "🦆",
    "swan": "🦢",
    "swans": "🦢",
    "owl": "🦉",
    "owls": "🦉",
    "flamingo": "🦩",
    "flamingos": "🦩",
    "peacock": "🦚",
    "peacocks": "🦚",
    "parrot": "🦜",
    "parrots": "🦜",
    "frog": "🐸",
    "frogs": "🐸",
    "crocodile": "🐊",
    "crocodiles": "🐊",
    "turtle": "🐢",
    "turtles": "🐢",
    "lizard": "🦎",
    "lizards": "🦎",
    "snake": "🐍",
    "snakes": "🐍",
    "dragon": "🐉",
    "dragons": "🐉",
    "dinosaur": "🦕",
    "dinosaurs": "🦕",
    "whale": "🐳",
    "whales": "🐳",
    "dolphin": "🐬",
    "dolphins": "🐬",
    "seal": "🦭",
    "seals": "🦭",
    "shark": "🦈",
    "sharks": "🦈",
    "octopus": "🐙",
    "octopuses": "🐙",
    "shell": "🐚",
    "shells": "🐚",
    "coral": "🪸",
    "corals": "🪸",
    "butterfly": "🦋",
    "butterflies": "🦋",
    "bug": "🐛",
    "bugs": "🐛",
    "ant": "🐜",
    "ants": "🐜",
    "honeybee": "🐝",
    "honeybees": "🐝",
    "ladybug": "🐞",
    "ladybugs": "🐞",
    "cricket": "🦗",
    "crickets": "🦗",
    "spider": "🕷️",
    "spiders": "🕷️",
    "scorpion": "🦂",
    "scorpions": "🦂",
    "microbe": "🦠",
    "microbes": "🦠",
    "bouquet": "💐",
    "bouquets": "💐",
    "tulip": "🌷",
    "tulips": "🌷",
    "rose": "🌹",
    "roses": "🌹",
    "wilted flower": "🥀",
    "sunflower": "🌻",
    "sunflowers": "🌻",
    "blossom": "🌼",
    "blossoms": "🌼",
    "herb": "🌿",
    "herbs": "🌿",
    "shamrock": "☘️",
    "shamrocks": "☘️",
    "maple leaf": "🍁",
    "maple leaves": "🍁",
    "fallen leaf": "🍂",
    "fallen leaves": "🍂",
    "leaf fluttering": "🍃",
    "leaves fluttering": "🍃",
    "mushroom": "🍄",
    "mushrooms": "🍄",
    "cactus": "🌵",
    "cacti": "🌵",
    "palm tree": "🌴",
    "palm trees": "🌴",
    "evergreen tree": "🌲",
    "evergreen trees": "🌲",
    "deciduous tree": "🌳",
    "deciduous trees": "🌳",
    "christmas": "🎄",
    "mountain": "⛰️",
    "mountains": "⛰️",
    "volcano": "🌋",
    "volcanoes": "🌋",
    "desert": "🏜️",
    "deserts": "🏜️",
    "island": "🏝️",
    "islands": "🏝️",
    "national park": "🏞️",
    "national parks": "🏞️",
    "globe": "🌍",
    "earth": "🌎",
    "world": "🌏",
    "map": "🗺️",
    "maps": "🗺️",
    "japan": "🗾",
    "compass": "🧭",
    "compasses": "🧭",
    "snowcapped mountain": "🏔️",
    "snowcapped mountains": "🏔️",
    "camping": "🏕️",
    "beach": "🏖️",
    "beaches": "🏖️",
    "building construction": "🏗️",
    "houses": "🏘️",
    "city": "🏙️",
    "cities": "🏙️",
    "cityscapes": "🏙️",
    "derelict house": "🏚️",
    "derelict houses": "🏚️",
    "classical building": "🏛️",
    "classical buildings": "🏛️",
    "factory": "🏭",
    "factories": "🏭",
    "brick": "🧱",
    "bricks": "🧱",
    "rock": "🪨",
    "rocks": "🪨",
    "wood": "🪵",
    "woods": "🪵",
    "hut": "🛖",
    "huts": "🛖",
    "stadium": "🏟️",
    "stadiums": "🏟️",
    "ferris wheel": "🎡",
    "ferris wheels": "🎡",
    "roller coaster": "🎢",
    "roller coasters": "🎢",
    "carousel horse": "🎠",
    "carousel horses": "🎠",
    "fountain": "⛲",
    "fountains": "⛲",
    "tent": "⛺",
    "tents": "⛺",
    "foggy": "🌁",
    "night": "🌃",
    "sunrise": "🌅",
    "sunrises": "🌅",
    "sunset": "🌇",
    "sunsets": "🌇",
    "rainbow": "🌈",
    "rainbows": "🌈",
    "wave": "🌊",
    "waves": "🌊",
    "tornado": "🌪️",
    "tornados": "🌪️",
    "typhoon": "🌀",
    "typhoons": "🌀",
    "hurricane": "🌀",
    "hurricanes": "🌀",
    "fog": "🌫️",
    "wind face": "🌬️",
    "wind faces": "🌬️",
    "blowing wind": "🌬️",
    "hot spring": "♨️",
    "hot springs": "♨️",
    "thermometer": "🌡️",
    "thermometers": "🌡️",
    "drop": "💧",
    "drops": "💧",
    "sweat droplets": "💦",
    "ice": "🧊",
    "snowflake": "❄️",
    "snowflakes": "❄️",
    "snowman": "☃️",
    "snowmen": "☃️",
    "snowman without snow": "⛄",
    "comet": "☄️",
    "comets": "☄️",
    "fire": "🔥",
    "flames": "🔥",
    "jack-o-lantern": "🎃",
    "jack-o-lanterns": "🎃",
    "fireworks": "🎆",
    "sparkler": "🎇",
    "sparklers": "🎇",
    "firecracker": "🧨",
    "firecrackers": "🧨",
    "sparkles": "✨",
    "balloon": "🎈",
    "balloons": "🎈",
    "party": "🎉",
    "parties": "🎉",
    "confetti ball": "🎊",
    "confetti balls": "🎊",
    "tanabata tree": "🎋",
    "tanabata trees": "🎋",
    "pine": "🎍",
    "pine": "🎍",
    "japanese dolls": "🎎",
    "carp streamer": "🎏",
    "carp streamers": "🎏",
    "wind chime": "🎐",
    "wind chimes": "🎐",
    "moon viewing ceremony": "🎑",
    "ribbon": "🎀",
    "ribbons": "🎀",
    "gift": "🎁",
    "gifts": "🎁",
    "present": "🎁",
    "presents": "🎁",
    "ticket": "🎫",
    "tickets": "🎫",
    "medal": "🎖️",
    "medals": "🎖️",
    "trophy": "🏆",
    "trophies": "🏆",
    "1st": "🥇",
    "2nd": "🥈",
    "3rd": "🥉",
    "first": "🥇",
    "second": "🥈",
    "third": "🥉",
    "ball": "⚽",
    "balls": "⚽",
    "baseball": "⚾",
    "baseballs": "⚾",
    "softball": "🥎",
    "softballs": "🥎",
    "basketball": "🏀",
    "basketballs": "🏀",
    "volleyball": "🏐",
    "volleyballs": "🏐",
    "american": "🏈",
    "american": "🏈",
    "rugby": "🏉",
    "rugby": "🏉",
    "tennis": "🎾",
    "tennis balls": "🎾",
    "flying disc": "🥏",
    "flying discs": "🥏",
    "bowling": "🎳",
    "cricket": "🏏",
    "cricket": "🏏",
    "hockey": "🏑",
    "ice hockey": "🏒",
    "lacrosse": "🥍",
    "ping pong": "🏓",
    "badminton": "🏸",
    "boxing": "🥊",
    "boxing": "🥊",
    "goal net": "🥅",
    "goal nets": "🥅",
    "flag in hole": "⛳",
    "flags in hole": "⛳",
    "skate": "⛸️",
    "skates": "⛸️",
    "fishing": "🎣",
    "diving": "🤿",
    "running": "🎽",
    "skis": "🎿",
    "sled": "🛷",
    "sleds": "🛷",
    "curling stone": "🥌",
    "curling stones": "🥌",
    "bullseye": "🎯",
    "bullseyes": "🎯",
    "yo-yo": "🪀",
    "yo-yos": "🪀",
    "kite": "🪁",
    "kites": "🪁",
    "water pistol": "🔫",
    "water pistols": "🔫",
    "pool 8 ball": "🎱",
    "pool 8 balls": "🎱",
    "crystal ball": "🔮",
    "crystal balls": "🔮",
    "wand": "🪄",
    "wands": "🪄",
    "game": "🎮",
    "games": "🎮",
    "joystick": "🕹️",
    "joysticks": "🕹️",
    "slot machine": "🎰",
    "slot machines": "🎰",
    "die": "🎲",
    "dice": "🎲",
    "puzzle": "🧩",
    "puzzle": "🧩",
    "teddy": "🧸",
    "teddy": "🧸",
    "spade suit": "♠️",
    "heart suit": "♥️",
    "diamond suit": "♦️",
    "club suit": "♣️",
    "pawn": "♟️",
    "pawns": "♟️",
    "joker": "🃏",
    "jokers": "🃏",
    "mahjong": "🀄",
    "mahjong": "🀄",
    "flower playing cards": "🎴",
    "arts": "🎭",
    "art": "🎭",
    "picture": "🖼️",
    "pictures": "🖼️",
    "paint": "🎨",
    "paints": "🎨",
    "painter": "🎨",
    "thread": "🧵",
    "threads": "🧵",
    "needle": "🪡",
    "needles": "🪡",
    "yarn": "🧶",
    "yarns": "🧶",
    "knot": "🪢",
    "knots": "🪢",
    "mending heart": "💖",
    "mending hearts": "💖",
    "heart on fire": "❤️",
    "hearts on fire": "❤️",
    "face with spiral eyes": "😵",
    "faces with spiral eyes": "😵",
    "face in clouds": "😶",
    "faces in clouds": "😶",
    "face exhaling": "😮",
    "faces exhaling": "😮",
    "face with peeking eye": "👀",
    "faces with peeking eye": "👀",
    "saluting face": "🫡",
    "saluting faces": "🫡",
    "dotted line face": "👤",
    "dotted line faces": "👤",
    "face holding back tears": "😢",
    "faces holding back tears": "😢",
    "right": "👉",
    "left": "👈",
    "palm down hand": "🖐️",
    "palm down hands": "🖐️",
    "palm up hand": "🤲",
    "palm up hands": "🤲",
    "handshake": "🤝",
    "handshakes": "🤝",
    "heart hands": "🤗",
    "biting": "😬",
    "bite": "😬",
    "crown": "👑",
    "crowns": "👑",
    "pregnant": "🤰",
    "pregnant": "🤰",
    "troll": "👹",
    "trolls": "👹",
    "coral": "🐚",
    "corals": "🐚",
    "lotus": "🌸",
    "lotuses": "🌸",
    "nest": "🪹",
    "nests": "🪹",
    "egg": "🐣",
    "eggs": "🐣",
    "beans": "🫘",
    "liquid": "🍶",
    "jar": "🏺",
    "jars": "🏺",
    "slide": "🛝",
    "slides": "🛝",
    "wheel": "🛞",
    "wheels": "🛞",
    "buoy": "🛟",
    "buoys": "🛟",
    "hamsa": "🧿",
    "hamsas": "🧿",
    "mirror ball": "🕺",
    "mirror balls": "🕺",
    "battery": "🔋",
    "batteries": "🔋",
    "crutch": "🦯",
    "crutches": "🦯",
    "x-ray": "🩻",
    "x-rays": "🩻",
    "bubbles": "🫧",
    "card": "💳",
    "cards": "💳",
    "equals": "=",
    "equal": "=",
    "wireless": "📶",
    "khanda": "🕉️",
    "maracas": "🎵",
    "flute": "🎼",
    "flutes": "🎼",
    "hyacinth": "🌺",
    "hyacinths": "🌺",
    "jellyfish": "🐙",
    "jellyfishes": "🐙",
    "wing": "🦅",
    "wings": "🦅",
    "goose": "🦢",
    "geese": "🦢",
    "moose": "🦌",
    "donkey": "🐴",
    "donkeys": "🐴",
    "bird": "🐦",
    "birds": "🐦",
    "phoenix": "🦅",
    "phoenixes": "🦅",
    "ginger": "🥔",
    "pea pod": "🫛",
    "pea pods": "🫛",
    "folding hand fan": "🌬️",
    "folding hand fans": "🌬️",
    "hair pick": "🧼",
    "hair picks": "🧼",
}

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def bert_synonym_replacement(word_list, num_replacements, max_depth=3):
    def get_synonyms(word):
        url = f"http://api.conceptnet.io/c/en/{word}?rel=/r/Synonym&limit=10"
        response = requests.get(url)
        data = response.json()
        synonyms = [edge['end']['label'] for edge in data['edges']
                    if edge['rel']['label'] == 'Synonym' and edge['end']['language'] == 'en']
        return list(set(synonyms))

    def get_sentence_embedding(sentence):
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

    def is_similar(word1, word2):
        if word1.lower() in word2.lower() or word2.lower() in word1.lower():
            return True
        if abs(len(word1) - len(word2)) <= 1:
            matcher = SequenceMatcher(None, word1.lower(), word2.lower())
            return matcher.ratio() > 0.8
        return False

    def get_valid_synonym(word, synonyms, context):
        word_frequency = wordfreq.word_frequency(word, 'en')
        original_embedding = get_sentence_embedding(context)
        sorted_synonyms = sorted(synonyms, key=lambda x: wordfreq.word_frequency(x, 'en'), reverse=True)
        
        for i, synonym in enumerate(sorted_synonyms):
            if i >= max_depth:
                print(f"Reached maximum depth for '{word}'")
                return None
            
            if synonym.lower() == word.lower():
                print(f"Skipped '{word}' and '{synonym}' (same word)")
                continue

            synonym_frequency = wordfreq.word_frequency(synonym, 'en')
            
            if synonym_frequency >= word_frequency * 0.01:
                if not is_similar(word, synonym):
                    new_context = context.replace(word, synonym)
                    new_embedding = get_sentence_embedding(new_context)
                    similarity = torch.cosine_similarity(original_embedding, new_embedding, dim=0)

                    print(f"Considering '{synonym}' for '{word}'")
                    print(f"Frequencies: Word={word_frequency}, Synonym={synonym_frequency}")
                    print(f"Similarity: {similarity.item()}")

                    if similarity > 0.95:  # Adjust this threshold as needed
                        return synonym
                else:
                    print(f"Skipped '{word}' and '{synonym}' (too similar)")
            else:
                print(f"Skipped '{synonym}' due to low frequency (Word: {word_frequency}, Synonym: {synonym_frequency})")
        
        return None

    sentence = " ".join(word_list)
    replacements_made = 0

    for i, word in enumerate(word_list):
        if replacements_made >= num_replacements:
            break

        synonyms = get_synonyms(word.lower())
        if synonyms:
            valid_synonym = get_valid_synonym(word, synonyms, sentence)
            if valid_synonym:
                word_list[i] = valid_synonym.replace('_', ' ')
                sentence = " ".join(word_list)  # Update the sentence
                replacements_made += 1
                print(f"Replaced '{word}' with '{valid_synonym}'")

    return word_list