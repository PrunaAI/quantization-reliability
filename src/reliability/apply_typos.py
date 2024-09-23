import copy
import random
import string
import nltk
from difflib import SequenceMatcher
from googletrans import Translator
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import spacy

import requests
from difflib import SequenceMatcher
import wordfreq

import emoji
from nltk.stem import PorterStemmer
import re

import os

# Set the custom download directory
nltk_data_dir = '/nfs/students/daro/data/nltk_data'

# Add the custom directory to NLTK's data path
nltk.data.path.append(nltk_data_dir)

# Download NLTK data if not already present
if not os.path.exists(os.path.join(nltk_data_dir, 'corpora', 'words')):
    nltk.download('all', download_dir=nltk_data_dir)

translator = Translator()

# Load spaCy model
nlp = spacy.load("en_core_web_md")

def load_file_to_dict(file_path):
    cmw_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            correct_word, incorrect_word = line.strip().split(':')
            cmw_dict[correct_word.strip()] = incorrect_word.strip()
    return cmw_dict

def word_random_phrase_translation(word_list, num_translations):
    """Translate random words or phrases to a random foreign language."""
    languages = ['es', 'fr', 'de', 'it', 'ru', 'zh-cn', 'ja']
    for _ in range(num_translations):
        if len(word_list) < 2:
            return word_list
        
        is_phrase = random.choice([True, False]) if len(word_list) > 2 else False
        
        try:
            if is_phrase:
                start_index = random.randint(0, len(word_list) - 2)
                phrase = ' '.join(word_list[start_index:start_index+2])
                lang = random.choice(languages)
                translated_phrase = translator.translate(phrase, dest=lang).text
                word_list[start_index:start_index+2] = translated_phrase.split()
            else:
                word_idx = random.randint(0, len(word_list) - 1)
                lang = random.choice(languages)
                translated_word = translator.translate(word_list[word_idx], dest=lang).text
                word_list[word_idx] = translated_word
        except Exception as e:
            print(f"Translation error: {e}")
    
    return word_list

def char_random_insertion(word, num_insertions):
    for _ in range(num_insertions):
        pos = random.randint(0, len(word))
        char_to_insert = random.choice(string.ascii_letters)
        word = word[:pos] + char_to_insert + word[pos:]
    return word

def char_random_deletion(word, num_deletions):
    for _ in range(num_deletions):
        if len(word) > 1:
            pos = random.randint(0, len(word) - 1)
            word = word[:pos] + word[pos+1:]
    return word

def char_random_replacement(word, num_replacements):
    for _ in range(num_replacements):
        if len(word) > 0:
            pos = random.randint(0, len(word) - 1)
            if word[pos].lower() in keyboard_adjacency:
                replacement_char = random.choice(keyboard_adjacency[word[pos].lower()])
                word = word[:pos] + replacement_char + word[pos+1:]
    return word

def char_random_repetition(word, num_repetitions):
    for _ in range(num_repetitions):
        pos = random.randint(0, len(word))
        word = word[:pos] + word[pos-1:pos] + word[pos:]
    return word

def char_random_swapping(word, num_swaps):
    for _ in range(num_swaps):
        if len(word) > 1:
            pos = random.randint(0, len(word) - 2)
            word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
    return word

def word_apply_cmw(word_list, num_replacements, cmw_dict):
    def process_word(word):
        # Extract punctuation
        start_punct = re.match(r'^[^\w\s]+', word)
        end_punct = re.search(r'[^\w\s]+$', word)
        # Remove punctuation
        clean_word = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word)
        return clean_word, start_punct, end_punct

    def restore_punctuation(word, start_punct, end_punct):
        result = word
        if start_punct:
            result = start_punct.group() + result
        if end_punct:
            result = result + end_punct.group()
        return result

    misspellable_words = []
    for i, word in enumerate(word_list):
        clean_word, start_punct, end_punct = process_word(word)
        if clean_word.lower() in cmw_dict:
            misspellable_words.append((i, word, clean_word, start_punct, end_punct))

    num_replacements = min(num_replacements, len(misspellable_words))
    replaced_indices = set()

    for _ in range(num_replacements):
        if misspellable_words:
            index, original_word, clean_word, start_punct, end_punct = random.choice(misspellable_words)
            if index not in replaced_indices:
                misspelled_word = cmw_dict[clean_word.lower()]
                word_list[index] = restore_punctuation(misspelled_word, start_punct, end_punct)
                replaced_indices.add(index)
                misspellable_words = [w for w in misspellable_words if w[0] not in replaced_indices]

    return word_list

def char_random_letter_case(word, num_case_changes):
    for _ in range(num_case_changes):
        if len(word) > 0:
            pos = random.randint(0, len(word) - 1)
            word = word[:pos] + word[pos].swapcase() + word[pos+1:]
    return word

def word_synonym_replacement(word_list, num_replacements):
    def process_word(word):
        # Extract punctuation
        start_punct = re.match(r'^[^\w\s]+', word)
        end_punct = re.search(r'[^\w\s]+$', word)
        # Remove punctuation
        clean_word = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word)
        return clean_word, start_punct, end_punct

    def restore_punctuation(word, start_punct, end_punct):
        result = word
        if start_punct:
            result = start_punct.group() + result
        if end_punct:
            result = result + end_punct.group()
        return result

    def get_synonyms(word):
        url = f"http://api.conceptnet.io/c/en/{word}?rel=/r/Synonym&limit=1000"
        response = requests.get(url)
        data = response.json()
        synonyms = []
        for edge in data['edges']:
            if edge['rel']['label'] == 'Synonym' and edge['end']['language'] == 'en':
                synonyms.append(edge['end']['label'])
        return list(set(synonyms))

    def get_valid_synonym(word, synonyms, max_depth=3):
        word_frequency = wordfreq.word_frequency(word, 'en')
        sorted_synonyms = sorted(synonyms, key=lambda x: wordfreq.word_frequency(x, 'en'), reverse=True)
        for i, synonym in enumerate(sorted_synonyms):
            if i >= max_depth:
                print(f"Reached maximum depth for '{word}'")
                return None  # Stop if we've reached the maximum depth
            if synonym.lower() == word.lower():
                print(f"Skipped '{word}' and '{synonym}' (same word)")
                continue
            synonym_frequency = wordfreq.word_frequency(synonym, 'en')
            # Check if the synonym frequency is at least 1% of the original word frequency
            if synonym_frequency >= word_frequency * 0.01:
                if not is_similar(word, synonym):
                    print(f"Replaced '{word}' with '{synonym}'")
                    print(f"Frequencies: Word={word_frequency}, Synonym={synonym_frequency}")
                    return synonym
                else:
                    print(f"Skipped '{word}' and '{synonym}' (too similar)")
            else:
                print(f"Skipped '{synonym}' due to low frequency (Word: {word_frequency}, Synonym: {synonym_frequency})")
        return None

    def is_similar(word1, word2):
        # Check if one word is contained within the other
        if word1.lower() in word2.lower() or word2.lower() in word1.lower():
            return True
        # Check if the words differ by only one character
        if abs(len(word1) - len(word2)) <= 1:
            matcher = SequenceMatcher(None, word1.lower(), word2.lower())
            return matcher.ratio() > 0.8  # Adjust this threshold as needed
        return False

    replacements_made = 0
    replaced_indices = set()

    for i, word in enumerate(word_list):
        if replacements_made >= num_replacements:
            break
        if i in replaced_indices:
            continue

        clean_word, start_punct, end_punct = process_word(word)
        synonyms = get_synonyms(clean_word)

        if synonyms:
            valid_synonym = get_valid_synonym(clean_word, synonyms)
            if valid_synonym:
                word_list[i] = restore_punctuation(valid_synonym.replace('_', ' '), start_punct, end_punct)
                replaced_indices.add(i)
                replacements_made += 1

    return word_list

def word_context_aware_insertion(word_list, num_insertions):
    def process_word(word):
        # Extract punctuation
        start_punct = re.match(r'^[^\w\s]+', word)
        end_punct = re.search(r'[^\w\s]+$', word)
        # Remove punctuation
        clean_word = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word)
        return clean_word, start_punct, end_punct

    def restore_punctuation(word, start_punct, end_punct):
        result = word
        if start_punct:
            result = start_punct.group() + result
        if end_punct:
            result = result + end_punct.group()
        return result

    # Load pre-trained RoBERTa model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForMaskedLM.from_pretrained("roberta-base")
    model.eval()

    # Process all words to handle punctuation
    processed_words = [process_word(word) for word in word_list]
    clean_word_list = [word[0] for word in processed_words]

    for _ in range(num_insertions):
        # Choose a random position to insert a word
        insert_position = random.randint(1, len(clean_word_list) - 1)

        # Create a masked sentence for prediction
        masked_sentence = clean_word_list[:insert_position] + [tokenizer.mask_token] + clean_word_list[insert_position:]
        masked_sentence = " ".join(masked_sentence)

        # Tokenize and get model predictions
        inputs = tokenizer(masked_sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted token
        mask_token_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]
        predicted_token_id = outputs.logits[0, mask_token_index].argmax(axis=-1)
        predicted_token = tokenizer.decode(predicted_token_id)

        # Insert the predicted word if it's not empty
        if predicted_token.strip() != "":
            clean_word_list.insert(insert_position, predicted_token.strip())
            processed_words.insert(insert_position, (predicted_token.strip(), None, None))

    # Restore punctuation
    result_word_list = [restore_punctuation(word, start_punct, end_punct) 
                        for word, start_punct, end_punct in processed_words]

    return result_word_list

def char_add_noise_characters(word, num_noises):
    for _ in range(num_noises):
        pos = random.randint(0, len(word))
        noise_char = random.choice(string.punctuation + string.digits)
        word = word[:pos] + noise_char + word[pos:]
    return word

def word_repeat_key_words(word_list, num_repetitions):
    for _ in range(num_repetitions):
        if word_list:
            word_idx = random.randint(0, len(word_list) - 1)
            word_list.insert(word_idx, word_list[word_idx])
    return word_list

def char_random_char_substitution(word, num_substitutions):
    for _ in range(num_substitutions):
        if len(word) > 0:
            pos = random.randint(0, len(word) - 1)
            if word[pos].upper() in char_map:
                word = word[:pos] + char_map[word[pos].upper()] + word[pos+1:]
    return word

def word_internet_slang_insertion(word_list, num_insertions):
    for _ in range(num_insertions):
        pos = random.randint(0, len(word_list))
        word_list.insert(pos, random.choice(internet_slang))
    return word_list

def load_emoji_mappings():
    emoji_mappings = {}
    ps = PorterStemmer()
    
    for emoji_code, emoji_data in emoji.EMOJI_DATA.items():
        if 'en' in emoji_data:
            description = emoji_data['en'].replace(':', '')
            words = re.findall(r'\w+', description)
            for word in words:
                stemmed_word = ps.stem(word.lower())
                if stemmed_word not in emoji_mappings:
                    emoji_mappings[stemmed_word] = []
                emoji_mappings[stemmed_word].append(emoji_code)
    
    return emoji_mappings

def word_emoji_substitution(word_list, num_substitutions):
    emoji_mappings = load_emoji_mappings()
    ps = PorterStemmer()

    def find_matching_emoji(word):
        # Extract punctuation
        start_punct = re.match(r'^[^\w\s]+', word)
        end_punct = re.search(r'[^\w\s]+$', word)
        
        # Remove punctuation and stem the word
        clean_word = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', word)
        stemmed_word = ps.stem(clean_word.lower()).lower()

        for key in emoji_mappings.keys():
            if (f"_{stemmed_word}_" in f"_{key}_" and key.count('_') <= 1) or f"{stemmed_word}s" == key or stemmed_word == key:
                emoji_code = random.choice(emoji_mappings[key])
                
                # Add punctuation back to the emoji
                result = emoji.emojize(emoji_code)
                if start_punct:
                    result = start_punct.group() + result
                if end_punct:
                    result = result + end_punct.group()
                
                return result
        return None

    # Pre-process all words to find potential emoji matches
    substitutable_words = []
    for i, word in enumerate(word_list):
        matching_emoji = find_matching_emoji(word)
        if matching_emoji:
            substitutable_words.append((i, word, matching_emoji))

    # Perform substitutions
    substituted_indices = set()
    total_substitutions = 0
    while total_substitutions < num_substitutions and substitutable_words:
        index, word, emoji_replacement = random.choice(substitutable_words)
        if index not in substituted_indices:
            word_list[index] = emoji_replacement
            substituted_indices.add(index)
            total_substitutions += 1
            print(f"Substituted '{word}' with '{emoji_replacement}'")
        substitutable_words = [w for w in substitutable_words if w[0] not in substituted_indices]

    return word_list

def word_remove_punctuation(words, num_modifications):
    query = ' '.join(words)
    punctuation_positions = [i for i, char in enumerate(query) if char in string.punctuation]
    
    if not punctuation_positions:
        return query.split()  # No punctuation to remove
    
    query_chars = list(query)
    
    for _ in range(min(num_modifications, len(punctuation_positions))):
        if not punctuation_positions:
            break
        
        remove_index = random.choice(punctuation_positions)
        print(f"Removed punctuation: {query[remove_index]}")
        query_chars[remove_index] = ''
        punctuation_positions = [pos for pos in punctuation_positions if pos != remove_index]
    
    words = ''.join(query_chars).split()
    return words

def word_fill_word_deletion(words, num_modifications):
    stop_word_indices = [i for i, word in enumerate(words) if word.lower() in stop_words]
    
    if not stop_word_indices:
        return query.split()  # No stop words to remove
    
    for _ in range(min(num_modifications, len(stop_word_indices))):
        if not stop_word_indices:
            break
        
        remove_index = random.choice(stop_word_indices)
        print(f"Removed stop word: {words[remove_index]}")
        words[remove_index] = ''
        stop_word_indices = [i for i in stop_word_indices if i != remove_index]
    
    words = [word for word in words if word]
    return words

def apply_typo_modifications(query, typo_dict, taxonomy=[]):
    query = copy.deepcopy(query)
    cmw_file_path = 'data/cmw_v2.txt'
    cmw_dict = load_file_to_dict(cmw_file_path)
    words = query.split()
    
    for mod_type, num_modifications in typo_dict.items():
        if num_modifications > 0:
            if mod_type == 'word_CMW':
                words = word_apply_cmw(words, num_modifications, cmw_dict)
            elif mod_type == 'word_synonym':
                words = word_synonym_replacement(words, num_modifications)
            elif mod_type in ['char_noise', 'char_char_substitution', 'char_insertion', 'char_deletion', 'char_replacement', 'char_repetition', 'char_swapping', 'char_LCC']:
                indices = random.sample(range(len(words)), min(num_modifications, len(words)))
                for idx in indices:
                    if mod_type == 'char_insert_noise':
                        words[idx] = char_add_noise_characters(words[idx], 1)
                    elif mod_type == 'char_substitution':
                        words[idx] = char_random_char_substitution(words[idx], 1)
                    elif mod_type == 'char_insertion':
                        words[idx] = char_random_insertion(words[idx], 1)
                    elif mod_type == 'char_deletion':
                        words[idx] = char_random_deletion(words[idx], 1)
                    elif mod_type == 'char_replacement':
                        words[idx] = char_random_replacement(words[idx], 1)
                    elif mod_type == 'char_repetition':
                        words[idx] = char_random_repetition(words[idx], 1)
                    elif mod_type == 'char_swapping':
                        words[idx] = char_random_swapping(words[idx], 1)
                    elif mod_type == 'char_LCC':
                        words[idx] = char_random_letter_case(words[idx], 1)
            elif mod_type == 'word_emoji':
                words = word_emoji_substitution(words, num_modifications)
            elif mod_type == 'word_internet_slang':
                words = word_internet_slang_insertion(words, num_modifications)
            elif mod_type == 'word_phrase_translation':
                words = word_random_phrase_translation(words, num_modifications)
            elif mod_type == 'word_repeat':
                words = word_repeat_key_words(words, num_modifications)
            elif mod_type == 'word_context_aware_insertion':
                words = word_context_aware_insertion(words, num_modifications)
            elif mod_type == 'word_remove_punctuation':
                words = word_remove_punctuation(words, num_modifications)
            elif mod_type == 'word_keyword_only':
                words = word_fill_word_deletion(words, num_modifications)
            else:
                print(f"Invalid modification type: {mod_type}")
    
    modified_query = ' '.join(words)
    
    return modified_query