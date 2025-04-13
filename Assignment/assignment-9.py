                          #ASSIGNMENT-9
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, LancasterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Q1. Write a unique paragraph (5-6 sentences) about your favorite topic (e.g., sports,
# technology, food, books, etc.).
# 1. Convert text to lowercase and remove punctuation.
# 2. Tokenize the text into words and sentences.
# 3. Remove stopwords (using NLTK's stopwords list).
# 4. Display word frequency distribuƟon (excluding stopwords).

text = """Technology is evolving rapidly and transforming the way we live. 
From smartphones to artificial intelligence, innovations are everywhere. 
It enhances productivity, connects people globally, and solves real-world problems. 
The tech industry continues to grow with exciting new developments each year. 
Staying updated with technological trends is essential in today's fast-paced world."""

text_lower = text.lower()
text_clean = text_lower.translate(str.maketrans('', '', string.punctuation))
print("Cleaned Text (Lowercase & Punctuation Removed):")
print(text_clean)
print("\n")

words = word_tokenize(text_clean)
sentences = sent_tokenize(text)
print("Tokenized Words:")
print(words)
print("\n")
print("Tokenized Sentences:")
print(sentences)
print("\n")

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]
print("Filtered Tokens (Stopwords Removed):")
print(filtered_words)
print("\n")

freq_dist = FreqDist(filtered_words)
print("Word Frequency Distribution (Excluding Stopwords):")
for word, freq in freq_dist.items():
    print(f"{word}: {freq}")


# Q2: Stemming and Lemmatization
# 1. Take the tokenized words from Question 1 (after stopword removal).
# 2. Apply stemming using NLTK's PorterStemmer and LancasterStemmer.
# 3. Apply lemmatization using NLTK's WordNetLemmatizer.
# 4. Compare and display results of both techniques.

porter = PorterStemmer()
lancaster = LancasterStemmer()
lemmatizer = WordNetLemmatizer()
porter_stems = [porter.stem(word) for word in filtered_words]
lancaster_stems = [lancaster.stem(word) for word in filtered_words]
lemmas = [lemmatizer.lemmatize(word) for word in filtered_words]
print("\nPorter Stemmer:", porter_stems)
print("Lancaster Stemmer:", lancaster_stems)
print("Lemmatized Words:", lemmas)


# Q3. Regular Expressions and Text Spliƫng
# 1. Take their original text from Question 1.
# 2. Use regular expressions to:
# a. Extract all words with more than 5 letters.
# b. Extract all numbers (if any exist in their text).
# c. Extract all capitalized words.
# 3. Use text spliƫng techniques to:
# a. Split the text into words containing only alphabets (removing digits and special
# characters).
# b. Extract words starting with a vowel.

original_text = text
long_words = re.findall(r'\b\w{6,}\b', original_text)
numbers = re.findall(r'\b\d+\b', original_text)
capitalized = re.findall(r'\b[A-Z][a-z]*\b', original_text)
alphabet_words = re.findall(r'\b[a-zA-Z]+\b', original_text)
vowel_words = [word for word in alphabet_words if word[0].lower() in 'aeiou']

print("\nWords >5 letters:", long_words)
print("Numbers:", numbers)
print("Capitalized Words:", capitalized)
print("Alphabet-only Words:", alphabet_words)
print("Words starting with vowels:", vowel_words)


# Q4. Custom TokenizaƟon & Regex-based Text Cleaning
# 1. Take original text from Question 1.
# 2. Write a custom tokenization function that:
# a. Removes punctuation and special symbols, but keeps contractions (e.g.,
# "isn't" should not be split into "is" and "n't").
# b. Handles hyphenated words as a single token (e.g., "state-of-the-art" remains
# a single token).
# c. Tokenizes numbers separately but keeps decimal numbers intact (e.g., "3.14"
# should remain as is).
# 3. Use Regex Substitutions (re.sub) to:
# a. Replace email addresses with '<EMAIL>' placeholder.
# b. Replace URLs with '<URL>' placeholder.
# c. Replace phone numbers (formats: 123-456-7890 or +91 9876543210) with
# '<PHONE>' placeholder.

def custom_tokenize(text):
    pattern = r"\b(?:[a-zA-Z0-9]+(?:['-][a-zA-Z0-9]+)*)\b|\d+\.\d+|\d+"
    return re.findall(pattern, text)
extended_text = """
Reach out via email at test@example.com or visit https://example.com for info. 
You can also call at 123-456-7890 or +91 9876543210.
Technology-driven tools like state-of-the-art systems aren't rare now."""
tokens = custom_tokenize(extended_text)
cleaned_text = re.sub(r'\S+@\S+', '<EMAIL>', extended_text)
cleaned_text = re.sub(r'https?://\S+', '<URL>', cleaned_text)
cleaned_text = re.sub(r'(\+91\s?\d{10}|\d{3}-\d{3}-\d{4})', '<PHONE>', cleaned_text)

print("\nCustom Tokens:", tokens)
print("\nCleaned Text with Regex Substitutions:\n", cleaned_text)
