import spacy

# Load English language model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "Barack Obama was the 44th President of the United States. He was born in Hawaii on August 4, 1961."

# Process the text with spaCy
doc = nlp(text)

# Print named entities in the text
print("Named Entities:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")
import spacy


class QueryProcessor:
    def __init__(self):
        # Load the spaCy language model
        self.nlp = spacy.load("en_core_web_sm")

    def process_query(self, query):
        # Lowercase the query
        query = query.lower()

        # Process the query with spaCy to perform NER
        doc = self.nlp(query)

        # Extract named entities from the query
        named_entities = [(ent.text, ent.label_) for ent in doc.ents]

        # You can use named entities to refine the query or adjust search parameters
        # For now, let's just print the named entities
        print("Named Entities in query:", named_entities)

        # Return the processed query
        return query


from nltk.corpus import wordnet


def expand_query(query):
    words = query.split()
    expanded_query = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            expanded_query.append(synonyms[0].lemmas()[0].name())
        else:
            expanded_query.append(word)
    return ' '.join(expanded_query)


query = 'example query'
expanded_query = expand_query(query)
print('Expanded Query:', expanded_query)

from flask import Flask, request

app = Flask(__name__)


@app.route('/feedback', methods=['POST'])
def feedback():
    user_feedback = request.form.get('feedback')
    # Store the feedback in a database or file
    with open('feedback.txt', 'a') as f:
        f.write(user_feedback + '\n')
    return 'Feedback received', 200


if __name__ == '__main__':
    app.run()

'''that uses Term Frequency-Inverse Document Frequency (TF-IDF) weighting and BM25 to rank documents based on query terms, and natural language processing (NLP) techniques such as Named Entity Recognition (NER) and Part-of-Speech (POS) tagging.
Required Libraries:

    Scikit-learn: For TF-IDF vectorization and cosine similarity calculation.
    RankBM25: For implementing the BM25 retrieval model.
    spaCy: For Named Entity Recognition (NER) and Part-of-Speech (POS) tagging
        TF-IDF weighting: The code uses the TfidfVectorizer from scikit-learn to calculate TF-IDF scores for documents and queries.
    BM25: The code uses the BM25Okapi from the rank-bm25 library to calculate BM25 similarity scores.
    Named Entity Recognition (NER): The code uses spaCy's English model (en_core_web_sm) to identify named entities in the documents and queries.
    Part-of-Speech (POS) Tagging: The code uses spaCy to tag the parts of speech of words in documents and queries.
       The preprocess_text function preprocesses the input text by removing punctuation and converting it to lowercase.
    The analyze_text function uses spaCy to perform Named Entity Recognition (NER) and Part-of-Speech (POS) tagging on the input text.
    The search_ir_system function preprocesses the query and transforms it using the TF-IDF vectorizer. It calculates cosine similarity scores using TF-IDF and BM25 similarity scores.
    The combined scores are calculated as the average of TF-IDF similarity scores and BM25 scores (you can experiment with different weighting).
    The function ranks the documents based on the combined scores and displays the results.
    '''


import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import spacy

# Load spaCy's English model for NER and POS tagging
nlp = spacy.load("en_core_web_sm")

# Sample documents
documents = [
    "I want to eat an apple.",
    "An apple a day keeps the doctor away.",
    "I enjoy eating fresh apples in the morning."
]


# Function to preprocess text (removes punctuation and converts to lowercase)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# Tokenize and preprocess documents
preprocessed_documents = [preprocess_text(doc) for doc in documents]

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)

# Initialize BM25
bm25 = BM25Okapi(preprocessed_documents)


# Function to perform NER and POS tagging using spaCy
def analyze_text(text):
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]
    pos_tags = [(token.text, token.pos_) for token in doc]
    return named_entities, pos_tags


# Function to search the IR system using a query
def search_ir_system(query):
    # Preprocess the query
    preprocessed_query = preprocess_text(query)

    # Perform NER and POS tagging on the query
    named_entities, pos_tags = analyze_text(query)
    print(f"Named Entities in query: {named_entities}")
    print(f"POS Tags in query: {pos_tags}")

    # Transform the query using the TF-IDF vectorizer
    query_vector = vectorizer.transform([preprocessed_query])

    # Calculate cosine similarity using TF-IDF
    tfidf_similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Calculate BM25 scores
    bm25_scores = bm25.get_scores(preprocessed_query.split())

    # Combine the scores for ranking (you may experiment with different weighting)
    combined_scores = (tfidf_similarity_scores + bm25_scores) / 2

    # Get the indices of documents sorted by combined scores
    sorted_indices = combined_scores.argsort()[::-1]

    # Display the ranked search results
    print("\nSearch Results (ranked by relevance):")
    for rank, doc_index in enumerate(sorted_indices):
        print(f"Rank {rank + 1}: Document {doc_index + 1}: {documents[doc_index]}")


# Example query
query = "I want an apple"
search_ir_system(query)

'''To implement phrase matching and adjust the retrieval model to handle phrases, you can modify the information retrieval (IR) system to use n-gram tokenization and search for documents that contain the exact phrases. This approach involves considering not just single words (unigrams) but also word pairs (bigrams) and trigrams to preserve the context and order of words in the query.

Here are the key steps to implement phrase matching:

    Adjust the Tokenization: Modify the tokenizer to handle n-grams (phrases) in addition to unigrams. This can be done using the TfidfVectorizer from scikit-learn with an ngram_range parameter.

    Search for Exact Phrases: After tokenization, you can match the query as a phrase or phrases against the documents.

    Combine Scores for Ranking: Calculate similarity scores using cosine similarity, BM25, or other retrieval models. Combine the scores to rank the documents.

    Display the Top Results: Display the top-ranked documents based on their similarity to the query.
        CountVectorizer: Handles n-grams (phrases) in addition to unigrams, using an n-gram range of (1, 2).
    Phrase Queries: query_vector and document_vectors represent the query and documents in the vector space model, considering unigrams and bigrams.
    BM25: The BM25Okapi instance calculates BM25 scores for the query.
    Search Results: The function search_ir_system combines the similarity scores from cosine similarity and BM25, sorts them by relevance, and displays the top-ranked results.

You can adjust the ngram_range in CountVectorizer based on your requirements to consider larger phrases (e.g., trigrams) if necessary.
    '''
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import spacy

# Load spaCy's English model for NER and POS tagging
nlp = spacy.load("en_core_web_sm")

# Sample documents
documents = [
    "I want to eat an apple.",
    "An apple a day keeps the doctor away.",
    "I enjoy eating fresh apples in the morning."
]


# Function to preprocess text (removes punctuation and converts to lowercase)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# Tokenize and preprocess documents
preprocessed_documents = [preprocess_text(doc) for doc in documents]

# Create a CountVectorizer with n-gram range for phrase matching
vectorizer = CountVectorizer(ngram_range=(1, 2))

# Fit the vectorizer on the preprocessed documents and transform them
document_vectors = vectorizer.fit_transform(preprocessed_documents)

# Initialize BM25
bm25 = BM25Okapi(preprocessed_documents)


# Function to perform NER and POS tagging using spaCy
def analyze_text(text):
    doc = nlp(text)
    named_entities = [ent.text for ent in doc.ents]
    pos_tags = [(token.text, token.pos_) for token in doc]
    return named_entities, pos_tags


# Function to search the IR system using a query
def search_ir_system(query):
    # Preprocess the query
    preprocessed_query = preprocess_text(query)

    # Perform NER and POS tagging on the query
    named_entities, pos_tags = analyze_text(query)
    print(f"Named Entities in query: {named_entities}")
    print(f"POS Tags in query: {pos_tags}")

    # Transform the query using the vectorizer (handles n-grams)
    query_vector = vectorizer.transform([preprocessed_query])

    # Calculate cosine similarity using document vectors
    cosine_similarity_scores = cosine_similarity(query_vector, document_vectors).flatten()

    # Calculate BM25 scores
    bm25_scores = bm25.get_scores(preprocessed_query.split())

    # Combine the scores for ranking (you may experiment with different weighting)
    combined_scores = (cosine_similarity_scores + bm25_scores) / 2

    # Get the indices of documents sorted by combined scores
    sorted_indices = combined_scores.argsort()[::-1]

    # Display the ranked search results
    print("\nSearch Results (ranked by relevance):")
    for rank, doc_index in enumerate(sorted_indices):
        print(f"Rank {rank + 1}: Document {doc_index + 1}: {documents[doc_index]}")


# Example query
query = "I want an apple"
search_ir_system(query)


def dynamic_ranking(similarity_scores, user_feedback):
    # Adjust ranking based on user feedback or contextual factors
    if user_feedback == 'positive':
        return similarity_scores * 1.5  # Boost relevance scores
    elif user_feedback == 'negative':
        return similarity_scores * 0.5  # Penalize relevance scores
    else:
        return similarity_scores  # No adjustment

# Apply dynamic ranking strategy
adjusted_scores = dynamic_ranking(similarity_scores, user_feedback)

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer with sparse matrix representation
tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', use_idf=True, smooth_idf=True)

# Vectorize documents using sparse matrix representation
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

from nltk.corpus import wordnet

def expand_query(query):
    expanded_query = []
    for word in query.split():
        synonyms = [synset.lemmas()[0].name() for synset in wordnet.synsets(word)]
        expanded_query.extend(synonyms)
    return ' '.join(expanded_query)

expanded_query = expand_query(query)
from nltk.corpus import wordnet

def expand_query(query):
    expanded_query = []
    for word in query.split():
        synonyms = [synset.lemmas()[0].name() for synset in wordnet.synsets(word)]
        expanded_query.extend(synonyms)
    return ' '.join(expanded_query)

expanded_query = expand_query(query)
vector_cache = {}

def vectorize_document(document):
    if document in vector_cache:
        return vector_cache[document]
    else:
        vector = vectorizer.transform([document])
        vector_cache[document] = vector
        return vector

document_vector = vectorize_document(document)

from nltk.corpus import wordnet

def suggest_query_refinement(query):
    synonyms = set()
    for word in query.split():
        for synset in wordnet.synsets(word):
            synonyms.update(synset.lemma_names())
    return list(synonyms)

import gensim.downloader as api

# Load pre-trained Word2Vec model
word2vec_model = api.load('word2vec-google-news-300')
