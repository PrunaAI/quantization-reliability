# Typo Modification Strategies

This document outlines various strategies used to modify input queries for testing the performance of language models under different types of distortions.

## 1. Random Insertion

### Description
Randomly inserts characters into words.

### Implementation
1. Choose a random position in the word.
2. Insert a random letter at that position.
3. Repeat for the specified number of insertions.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 insertion): "What is the capital of Framnce?"
- High intensity (3 insertions): "Whkat is the capitawl of Framnce?"

## 2. Random Deletion

### Description
Randomly removes characters from words.

### Implementation
1. Choose a random position in the word.
2. Remove the character at that position.
3. Repeat for the specified number of deletions.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 deletion): "What is the capital of Frnce?"
- High intensity (3 deletions): "Wht is the captal of Frnce?"

## 3. Random Replacement

### Description
Replaces characters with adjacent keys on a keyboard.

### Implementation
1. Choose a random position in the word.
2. Replace the character with a randomly chosen adjacent key.
3. Repeat for the specified number of replacements.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 replacement): "What is the capital of Ftance?"
- High intensity (3 replacements): "Whar ia the capital of Ftance?"

## 4. Random Repetition

### Description
Repeats characters in words.

### Implementation
1. Choose a random position in the word.
2. Duplicate the character at that position.
3. Repeat for the specified number of repetitions.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 repetition): "What is the capiital of France?"
- High intensity (3 repetitions): "Whhat is the capiital of Frannce?"

## 5. Random Swapping

### Description
Swaps adjacent characters in words.

### Implementation
1. Choose a random position in the word.
2. Swap the character at that position with the next character.
3. Repeat for the specified number of swaps.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 swap): "Waht is the capital of France?"
- High intensity (3 swaps): "Waht is hte capital fo Farnce?"

## 6. Common Misspelling Words (CMW)

### Description
Replaces words with common misspellings.

### Implementation
1. Load a dictionary of common misspellings.
2. For each word in the query, check if it has a common misspelling.
3. Replace the word with its misspelling if available.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 replacement): "What is the capitol of France?"
- High intensity (2 replacements): "Wat is the capitol of France?"

## 7. Random Letter Case Change (LCC)

### Description
Randomly changes the case of letters in words.

### Implementation
1. Choose a random position in the word.
2. Change the case of the character at that position.
3. Repeat for the specified number of case changes.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 change): "What is the capiTal of France?"
- High intensity (3 changes): "WhAt is the capiTal of FrAnce?"

## 8. Synonym Replacement

### Description
Replaces words with their synonyms or homophones.

### Implementation
1. For each word, find its synonyms or homophones using WordNet.
2. Replace the word with a randomly chosen synonym or homophone.
3. Repeat for the specified number of replacements.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 replacement): "What is the chief city of France?"
- High intensity (2 replacements): "What is the metropolis of the French Republic?"

## 9. Noise Character Insertion

### Description
Inserts non-alphabetic characters into words.

### Implementation
1. Choose a random position in the word.
2. Insert a random punctuation mark or digit at that position.
3. Repeat for the specified number of noise insertions.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 insertion): "What is the cap1ital of France?"
- High intensity (3 insertions): "Wh@at is the cap1ital of Fr&ance?"

## 10. Taxonomy Confusion

### Description
Adds related terms or concepts to the beginning of the query.

### Implementation
1. Maintain a list of related terms for each query.
2. Randomly select terms from this list.
3. Prepend the selected terms to the query.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 addition): "Paris. What is the capital of France?"
- High intensity (2 additions): "Paris. Lyon. What is the capital of France?"

## 11. Keyword Repetition

### Description
Repeats random words in the query.

### Implementation
1. Choose a random word from the query.
2. Insert that word at a random position in the query.
3. Repeat for the specified number of repetitions.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 repetition): "What is the capital the of France?"
- High intensity (2 repetitions): "What is the capital the of France capital?"

## 12. Character Substitution

### Description
Replaces characters with visually similar characters.

### Implementation
1. Maintain a dictionary of visually similar characters.
2. For each character in a word, check if it has a similar character.
3. Replace the character with its visually similar counterpart.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 substitution): "What is the capital of Frаnce?" (using Cyrillic 'а')
- High intensity (3 substitutions): "Whаt іs the cаpital of Frаnce?" (using Cyrillic 'а' and 'і')

## 13. Emoji Substitution

### Description
Replaces words with related emojis.

### Implementation
1. Maintain a dictionary of words and their corresponding emojis.
2. For each word in the query, check if it has a corresponding emoji.
3. Replace the word with its emoji if available.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 substitution): "What is the capital of 🇫🇷?"
- High intensity (2 substitutions): "What is the 🏛️ of 🇫🇷?"

## 14. Internet Slang Insertion

### Description
Inserts common internet slang terms into the query.

### Implementation
1. Maintain a list of common internet slang terms.
2. Choose a random position in the query.
3. Insert a randomly chosen slang term at that position.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 insertion): "What is the capital lol of France?"
- High intensity (2 insertions): "What is tbh the capital lol of France?"

## 15. Phrase Translation

### Description
Translates random words or phrases to a foreign language.

### Implementation
1. Choose a random word or phrase from the query.
2. Select a random target language.
3. Translate the chosen word or phrase to the target language.
4. Replace the original word or phrase with the translation.

### Example
Original query: "What is the capital of France?"
- Low intensity (1 translation): "What is the capitale of France?"
- High intensity (2 translations): "Qué es la capitale of France?"

## 16. Remove Punctuation and Capitalization

### Description
Removes all punctuation and converts the query to lowercase.

### Implementation
1. Convert the entire query to lowercase.
2. Remove all non-alphanumeric characters except spaces.

### Example
Original query: "What is the capital of France?"
- Applied: "what is the capital of france"

## 17. Keyword-Only Query

### Description
Removes all stop words from the query, leaving only keywords.

### Implementation
1. Maintain a list of common stop words.
2. Remove all words from the query that appear in the stop word list.

### Example
Original query: "What is the capital of France?"
- Applied: "capital France"

These strategies can be combined and applied with varying intensities to create a wide range of distorted queries for testing language model performance.
