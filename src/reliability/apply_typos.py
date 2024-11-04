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

import re

import os

from src.reliability import TYPO_TYPES
from src.reliability.constants import keyboard_adjacency, char_map, internet_slang, stop_words, load_common_emojis

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
    translator = Translator()
    available_indices = list(range(len(word_list)))
    total_translations = 0
    failed_translations = 0
    max_iterations = len(word_list) * 2  # Set a maximum number of iterations
    iterations = 0
    chosen_languages = []

    # Ensure num_translations doesn't exceed available words
    num_translations = min(num_translations, len(word_list))

    while total_translations < num_translations and len(available_indices) > 0 and iterations < max_iterations:
        iterations += 1
        is_phrase = random.choice([True, False]) if len(available_indices) > 1 else False
        word_idx = None
        try:
            # Choose language
            if len(chosen_languages) < 2:
                lang = random.choice(languages)
                if lang not in chosen_languages:
                    chosen_languages.append(lang)
            else:
                lang = random.choice(chosen_languages)

            if is_phrase and len(available_indices) > 1:
                start_index = random.choice(available_indices[:-1])
                if start_index + 1 in available_indices:
                    phrase = ' '.join(word_list[start_index:start_index+2])
                    translated_phrase = translator.translate(phrase, dest=lang).text
                    word_list[start_index:start_index+2] = translated_phrase.split()
                    available_indices.remove(start_index)
                    available_indices.remove(start_index + 1)
                    total_translations += 1
                else:
                    available_indices.remove(start_index)  # Remove problematic index
            else:
                word_idx = random.choice(available_indices)
                translated_word = translator.translate(word_list[word_idx], dest=lang).text
                word_list[word_idx] = translated_word
                available_indices.remove(word_idx)
                total_translations += 1

        except Exception as e:
            print(f"Translation error: {e}")
            failed_translations += 1
            if word_idx in available_indices:
                available_indices.remove(word_idx)  # Remove problematic index
            
        # Break if too many failed translations
        if failed_translations > len(word_list):
            print("Too many failed translations. Stopping.")
            break

    if iterations >= max_iterations:
        print("Maximum iterations reached. Stopping.")

    return word_list

def char_random_insertion(word, num_insertions):
    for _ in range(num_insertions):
        pos = random.randint(0, len(word))
        char_to_insert = random.choice(string.ascii_letters)
        if pos == 0:
            word = char_to_insert + word
        elif pos == len(word):
            word = word + char_to_insert
        else:
            word = word[:pos] + char_to_insert + word[pos:]
    return word

def char_random_deletion(word, num_deletions):
    for _ in range(num_deletions):
        if len(word) > 1:
            pos = random.randint(0, len(word) - 1)
            if pos == 0:
                word = word[1:]
            elif pos == len(word) - 1:
                word = word[:-1]
            else:
                word = word[:pos] + word[pos+1:]
    return word

def char_random_replacement(word, num_replacements):
    total_replacements = 0
    replaceable_chars = [i for i, char in enumerate(word) if char.lower() in keyboard_adjacency]
    
    while total_replacements < num_replacements and replaceable_chars:
        pos = random.choice(replaceable_chars)
        replacement_char = random.choice(keyboard_adjacency[word[pos].lower()])
        if pos == 0:
            word = replacement_char + word[1:]
        elif pos == len(word) - 1:
            word = word[:-1] + replacement_char
        else:
            word = word[:pos] + replacement_char + word[pos+1:]
        total_replacements += 1
        replaceable_chars.remove(pos)
    
    return word

def char_random_repetition(word, num_repetitions):
    for _ in range(num_repetitions):
        pos = random.randint(0, len(word)-1)
        if pos == 0:
            word = word[pos] + word
        elif pos == len(word)-1:
            word = word + word[pos]
        else:
            word = word[:pos] + word[pos] + word[pos:]
    return word

def char_random_swapping(word, num_swaps):
    total_swaps = 0
    available_positions = list(range(len(word) - 1))
    
    while total_swaps < num_swaps and len(available_positions) > 0:
        pos = random.choice(available_positions)
        if pos == 0:
            word = word[1] + word[0] + word[2:]
        elif pos == len(word) - 2:
            word = word[:-2] + word[-1] + word[-2]
        else:
            word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
        
        total_swaps += 1
        available_positions.remove(pos)
    
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

    total_replacements = 0
    while total_replacements < num_replacements and misspellable_words:
        index, original_word, clean_word, start_punct, end_punct = random.choice(misspellable_words)
        misspelled_word = cmw_dict[clean_word.lower()]
        word_list[index] = restore_punctuation(misspelled_word, start_punct, end_punct)
        total_replacements += 1
        misspellable_words = [w for w in misspellable_words if w[0] != index]

    return word_list

def char_random_letter_case(word, num_case_changes):
    total_changes = 0
    available_positions = list(range(len(word)))
    
    while total_changes < num_case_changes and available_positions:
        pos = random.choice(available_positions)
        
        if pos == 0:
            word = word[0].swapcase() + word[1:]
        elif pos == len(word) - 1:
            word = word[:-1] + word[-1].swapcase()
        else:
            word = word[:pos] + word[pos].swapcase() + word[pos+1:]
        
        total_changes += 1
        available_positions.remove(pos)
    
    return word

def word_synonym_replacement(word_list, num_replacements, cache_file_path='cache/synonym_cache.txt'):
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

    def load_cache(cache_file_path):
        cache = {}
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        word, syn = line.strip().split('\t')
                        if word not in cache:  # Only keep the first occurrence
                            cache[word] = syn
        except FileNotFoundError:
            pass
        return cache
    
    def save_to_cache(word, synonym, cache_file_path):
        with open(cache_file_path, 'a', encoding='utf-8') as f:
            f.write(f"{word}\t{synonym}\n")

    def get_synonyms(word):
        url = f"http://api.conceptnet.io/c/en/{word}?rel=/r/Synonym&limit=1000"
        response = requests.get(url)
        data = response.json()
        synonyms = []
        for edge in data['edges']:
            if edge['rel']['label'] == 'Synonym' and edge['end']['language'] == 'en':
                synonyms.append(edge['end']['label'])
        return list(set(synonyms))

    def get_valid_synonym(word, synonyms, cache, max_depth=3):
        # First check if word is in cache
        if word.lower() in cache:
            print(f"Using cached synonym for '{word}': '{cache[word.lower()]}'")
            return cache[word.lower()]

        word_frequency = wordfreq.word_frequency(word, 'en')
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
                    print(f"Replaced '{word}' with '{synonym}'")
                    print(f"Frequencies: Word={word_frequency}, Synonym={synonym_frequency}")
                    # Save valid pair to cache
                    save_to_cache(word.lower(), synonym, cache_file_path)
                    return synonym
                else:
                    print(f"Skipped '{word}' and '{synonym}' (too similar)")
            else:
                print(f"Skipped '{synonym}' due to low frequency (Word: {word_frequency}, Synonym: {synonym_frequency})")
        return None

    def is_similar(word1, word2):
        if word1.lower() in word2.lower() or word2.lower() in word1.lower():
            return True
        if abs(len(word1) - len(word2)) <= 1:
            matcher = SequenceMatcher(None, word1.lower(), word2.lower())
            return matcher.ratio() > 0.8
        return False

    # Load the cache at the start
    synonym_cache = load_cache(cache_file_path)

    words_with_synonyms = []
    for i, word in enumerate(word_list):
        clean_word, start_punct, end_punct = process_word(word)
        # First check cache
        if clean_word.lower() in synonym_cache:
            words_with_synonyms.append((i, word, clean_word, start_punct, end_punct, [synonym_cache[clean_word.lower()]]))
        else:
            synonyms = get_synonyms(clean_word)
            if synonyms and get_valid_synonym(clean_word, synonyms, synonym_cache):
                words_with_synonyms.append((i, word, clean_word, start_punct, end_punct, synonyms))

    replacements_made = 0
    attempted_indices = set()
    while replacements_made < num_replacements and words_with_synonyms and len(attempted_indices) < len(words_with_synonyms):
        available_words = [(i, w) for i, w in enumerate(words_with_synonyms) if i not in attempted_indices]
        if not available_words:
            break
            
        idx, (index, original_word, clean_word, start_punct, end_punct, synonyms) = random.choice(available_words)
        attempted_indices.add(idx)
        
        valid_synonym = get_valid_synonym(clean_word, synonyms, synonym_cache)
        if valid_synonym:
            word_list[index] = restore_punctuation(valid_synonym.replace('_', ' '), start_punct, end_punct)
            replacements_made += 1
            words_with_synonyms.pop(idx)

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
        masked_sentence = []
        if insert_position == len(clean_word_list) - 1:
            masked_sentence = clean_word_list + [tokenizer.mask_token]
        else:
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
        if pos == 0:
            word = noise_char + word
        elif pos == len(word):
            word = word + noise_char
        else:
            word = word[:pos] + noise_char + word[pos:]
    return word

def word_repeat_key_words(word_list, num_repetitions):
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

    # Process all words to handle punctuation
    processed_words = [process_word(word) for word in word_list]
    clean_word_list = [word[0] for word in processed_words]

    for _ in range(num_repetitions):
        if clean_word_list:
            word_idx = random.randint(0, len(clean_word_list) - 1)
            clean_word_list.insert(word_idx, clean_word_list[word_idx])
            processed_words.insert(word_idx, processed_words[word_idx])

    # Restore punctuation
    result_word_list = [restore_punctuation(word, start_punct, end_punct)
                        for word, start_punct, end_punct in processed_words]

    return result_word_list

def char_random_char_substitution(word, num_substitutions):
    total_substitutions = 0
    available_positions = list(range(len(word)))
    
    while total_substitutions < num_substitutions and available_positions:
        pos = random.choice(available_positions)
        if word[pos].upper() in char_map:
            if pos == 0:
                word = char_map[word[pos].upper()] + word[1:]
            elif pos == len(word) - 1:
                word = word[:-1] + char_map[word[pos].upper()]
            else:
                word = word[:pos] + char_map[word[pos].upper()] + word[pos+1:]
            total_substitutions += 1
        
        available_positions.remove(pos)
    
    return word

def word_internet_slang_insertion(word_list, num_insertions):
    for _ in range(num_insertions):
        pos = random.randint(0, len(word_list))
        word_list.insert(pos, random.choice(internet_slang))
    return word_list

def word_append_emoji(word_list, num_emojis):
    """
    Append random emojis to the end of the query text.
    
    Parameters:
    word_list (list): List of words in the query
    num_emojis (int): Number of emojis to append (intensity)
    
    Returns:
    list: Modified word list with emojis appended at the end
    """
    if num_emojis <= 0:
        return word_list
        
    # Load common emojis
    common_emojis = load_common_emojis()
    
    # Select random emojis
    selected_emojis = random.sample(common_emojis, min(num_emojis, len(common_emojis)))
    
    # Add emojis to the end of the word list
    word_list.extend(selected_emojis)
    
    print(f"Added {num_emojis} emojis at the end of the query: {' '.join(selected_emojis)}")
    
    return word_list

def word_remove_punctuation(words, num_modifications):
    query = ' '.join(words)
    punctuation_positions = [i for i, char in enumerate(query) if char in string.punctuation]
    
    if not punctuation_positions:
        return query.split()  # No punctuation to remove
    
    query_chars = list(query)
    modifications_made = 0
    
    while modifications_made < num_modifications and punctuation_positions:
        remove_index = random.choice(punctuation_positions)
        print(f"Removed punctuation: {query[remove_index]}")
        query_chars[remove_index] = ''
        punctuation_positions.remove(remove_index)
        modifications_made += 1
    
    words = ''.join(query_chars).split()
    return words

def word_fill_word_deletion(words, num_modifications):
    query = ' '.join(words)
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

def word_taxonomy_pos(words, num_modifications, taxonomy_list):
    answer = taxonomy_list[-1]  # The correct answer is the last item in the list
    query = ' '.join(words)
    for _ in range(num_modifications):
        query = f"{answer}. {query}"
    return query.split()

def word_taxonomy_neg(words, num_modifications, taxonomy_list):
    taxonomies = taxonomy_list[:-1]  # Exclude the correct answer
    query = ' '.join(words)
    for i in range(min(num_modifications, len(taxonomies))):
        query = f"{taxonomies[i]}. {query}"
    return query.split()

def apply_typo_modifications(query, typo_dict, taxonomy_list):
    query = copy.deepcopy(query)
    cmw_file_path = 'data/cmw_v2.txt'
    cmw_dict = load_file_to_dict(cmw_file_path)
    words = query.split()
    
    for mod_type, num_modifications in typo_dict.items():
        if mod_type not in TYPO_TYPES:
            raise ValueError(f"Invalid typo type: {mod_type}. Must be one of: {TYPO_TYPES}")
        if num_modifications > 0:
            if mod_type == 'word_CMW':
                words = word_apply_cmw(words, num_modifications, cmw_dict)
            elif mod_type == 'word_synonym':
                words = word_synonym_replacement(words, num_modifications)
            elif mod_type in ['char_insert_noise', 'char_substitution', 'char_insertion', 'char_deletion', 'char_replacement', 'char_repetition', 'char_swapping', 'char_LCC']:
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
                words = word_append_emoji(words, num_modifications)
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
            elif mod_type == 'word_taxonomy_pos':
                words = word_taxonomy_pos(words, num_modifications, taxonomy_list)
            elif mod_type == 'word_taxonomy_neg':
                words = word_taxonomy_neg(words, num_modifications, taxonomy_list)
            else:
                print(f"Invalid modification type: {mod_type}")
    
    modified_query = ' '.join(words)
    
    return modified_query