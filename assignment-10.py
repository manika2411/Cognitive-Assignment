                                    #ASSIGNMENT-10

import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Q1. Write a unique paragraph (5-6 sentences) about your favorite topic (e.g., sports,
# technology, food, books, etc.).
# 1. Convert text to lowercase and remove punctuaƟon using re.
# 2. Tokenize the text into words and sentences.
# 3. Split using split() and word_tokenize() and compare how Python split and NLTK’s
# word_tokenize() differ.
# 4. Remove stopwords (using NLTK's stopwords list).
# 5. Display word frequency distribuƟon (excluding stopwords).

my_paragraph = """I am absolutely fascinated by the world of Artificial Intelligence.
Its applications, ranging from self-driving cars to medical diagnosis, are truly revolutionary.
The rapid advancements in machine learning and deep learning are constantly reshaping industries.
I believe AI has the potential to solve some of humanity's most pressing challenges.
It's an exciting field, and I am eager to see what the future holds.
I enjoy reading research papers on AI."""

my_paragraph = my_paragraph.lower()
my_paragraph = re.sub(r'[^\w\s]', '', my_paragraph)

sentences = sent_tokenize(my_paragraph)
words = word_tokenize(my_paragraph)

split_words = my_paragraph.split()
print(f"Split with split(): {split_words}")
print(f"Split with word_tokenize(): {words}")

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in words if word not in stop_words]

freq_dist = FreqDist(filtered_words)
print("\nWord Frequency Distribution (excluding stopwords):")
for word, frequency in freq_dist.most_common(10):
    print(f"{word}: {frequency}")

freq_dist.plot(20, title='Word Frequency Distribution')
plt.show()

my_paragraph = """I am absolutely fascinated by the world of Artificial Intelligence.
Its applications, ranging from self-driving cars to medical diagnosis, are truly revolutionary.
"""

# Q2. Using the same paragraph from Q1:
# 1. Extract all words with only alphabets using re.findall()
# 2. Remove stop words using NLTK’s stopword list
# 3. Perform stemming with PorterStemmer
# 4. Perform lemmatization with WordNetLemmatizer
# 5. Compare the stemmed and lemmaƟzed outputs and explain when you’d prefer one over
# the other. 
alpha_words = re.findall(r'\b[a-zA-Z]+\b', my_paragraph)
print(f"\nWords with only alphabets: {alpha_words}")
filtered_alpha_words = [word for word in alpha_words if word not in stop_words]
porter_stemmer = PorterStemmer()
stemmed_words = [porter_stemmer.stem(word) for word in filtered_alpha_words]
print(f"Stemmed words: {stemmed_words}")
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_words = [wordnet_lemmatizer.lemmatize(word) for word in filtered_alpha_words]
print(f"Lemmatized words: {lemmatized_words}")
print("\nComparison of Stemming and Lemmatization:")
for i in range(len(filtered_alpha_words)):
    print(f"Original: {filtered_alpha_words[i]}, Stemmed: {stemmed_words[i]}, Lemmatized: {lemmatized_words[i]}")

# Q3. Choose 3 short texts of your own (e.g., different news headlines, product reviews).
# 1. Use CountVectorizer to generate the Bag of Words representation.
# 2. Use TfidfVectorizer to compute TF-IDF scores.
# 3. Print and interpret the top 3 keywords from each text using TF-IDF. 

texts = [
    "The new smartphone boasts an amazing camera and long battery life.",
    "This restaurant has delicious food and excellent service.",
    "The book provides a fascinating look into the history of artificial intelligence."
]

count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(texts)
bow_array = bow_matrix.toarray()
print("\nBag of Words representation:")
print(bow_array)
print(f"Features (words): {count_vectorizer.get_feature_names_out()}")

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
tfidf_array = tfidf_matrix.toarray()
print("\nTF-IDF representation:")
print(tfidf_array)
feature_names = tfidf_vectorizer.get_feature_names_out()

def get_top_keywords(tfidf_matrix, feature_names, top_n=3):
    keywords = []
    for row in tfidf_matrix:
        top_n_indices = row.argsort()[-top_n:][::-1]
        row_keywords = [feature_names[i] for i in top_n_indices]
        keywords.append(row_keywords)
    return keywords

top_keywords = get_top_keywords(tfidf_array, feature_names)
print("\nTop 3 keywords from each text:")
for i, text in enumerate(texts):
    print(f"Text {i+1}: {top_keywords[i]}")

# Q4. Write 2 short texts (4–6 lines each) describing two different technologies (e.g., AI vs
# Blockchain).
# 1. Preprocess and tokenize both texts.
# 2. Calculate:
# a. Jaccard Similarity using sets
# b. Cosine Similarity using TfidfVectorizer + cosine_similarity()
# c. Analyze which similarity metric gives beƩer insights in your case.

text1 = """
Artificial intelligence (AI) is rapidly transforming various aspects of our lives.
It involves the development of computer systems capable of performing tasks that typically require human intelligence.
Applications include automation, healthcare, finance.
Advancements in deep learning have boosted AI capabilities.
Ethical considerations are crucial in development of AI systems.
"""

text2 = """
Blockchain technology provides a decentralized and secure way to record transactions.
It operates on a distributed ledger where data is stored across multiple computers.
Blockchain has the potential to revolutionize supply chain management, voting, and digital identity.
Cryptocurrencies are one of the most known applications of blockchain.
Transparency and immutability are key advantages.
"""

def preprocess_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

tokens1 = preprocess_and_tokenize(text1)
tokens2 = preprocess_and_tokenize(text2)
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union else 0

jaccard_sim = jaccard_similarity(set(tokens1), set(tokens2))
print(f"\nJaccard Similarity: {jaccard_sim}")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([text1, text2])
cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
print(f"Cosine Similarity: {cosine_sim}")

# Q5. Write a short review for a product or service.
# 1. Use TextBlob or VADER to find polarity & subjectivity for each review.
# 2. Classify reviews into PosiƟve / Negative / Neutral.
# 3. Create a word cloud using the wordcloud library for all posiƟve reviews.

review = """This product is absolutely fantastic! The quality is excellent, and it works exactly as advertised.
I am extremely satisfied with my purchase. The customer service was also top-notch.
I highly recommend this to anyone looking for a reliable and effective solution.
However, the price was a bit high, but the value makes up for it."""

blob = TextBlob(review)
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity
print(f"\nPolarity: {polarity}, Subjectivity: {subjectivity}")
if polarity > 0.1:
    sentiment = "Positive"
elif polarity < -0.1:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

print(f"Sentiment: {sentiment}")
if sentiment == "Positive":
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(review)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Positive Review")
    plt.show()
else:
    print("\nNo positive review to generate word cloud.")

# Q6. Choose your own paragraph (~100 words) as training data.
# 1. Tokenize text using Tokenizer() from keras.preprocessing.text
# 2. Create input sequences and build a simple LSTM or Dense model
# 3. Train the model and generate 2–3 new lines of text starƟng from any seed word you
# provide. 

training_data = """The world of artificial intelligence is rapidly evolving.
Machine learning, a subset of AI, is enabling computers to learn from data without explicit programming.
Deep learning, a more advanced technique, utilizes neural networks with multiple layers to extract complex features.
These advancements have led to breakthroughs in various fields, including computer vision, natural language processing, and robotics.
The potential applications of AI are vast and transformative, promising to reshape industries and society.
As AI continues to progress, it is crucial to address the ethical considerations and ensure its responsible development.
"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([training_data])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in training_data.split("."):
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        n_gram_sequence = tokens[:i+1]
        input_sequences.append(n_gram_sequence)

max_seq_len = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_len, padding='pre')

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = np.array(y)

model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_seq_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=1)
seed_text = "artificial intelligence"
next_words = 10
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list, verbose=0), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word

print("\nGenerated text:")
print(seed_text)