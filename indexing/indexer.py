from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from query.query_processor import QueryProcessor
from storage.storage_manager import StorageManager


class Indexer:
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.vectorizer = None  # Hold a reference to the TF-IDF vectorizer
        self.document_vectors = None  # Hold document vectors after transformation
        self.storage_manager = StorageManager()
        self.query_processor = QueryProcessor()

    def index_documents(self, documents):
        # Create a TF-IDF vectorizer and fit it to the documents
        self.vectorizer = TfidfVectorizer()
        # Tokenize and preprocess documents
        preprocessed_documents = [self.query_processor.complete_process_query(doc.lower()) for doc in documents]
        processed_documents = [' '.join(doc) for doc in preprocessed_documents]
        print(preprocessed_documents)
        print(processed_documents)
        self.document_vectors = self.vectorizer.fit_transform(processed_documents)
        # # Save the vectorizer and document vectors
        self.storage_manager.save_vectorizer(self.vectorizer)
        self.storage_manager.save_document_vectors(self.document_vectors)
        for doc_id, document in enumerate(processed_documents):
            # document_words_list = self.query_processor.complete_process_query(document)
            # self.document_vectors = self.vectorizer.fit_transform(document_words_list)
            # self.storage_manager.save_vectorizer(self.vectorizer)
            # self.storage_manager.save_document_vectors(self.document_vectors)

            words = document.lower().split()  # Basic tokenization and lowercasing
            processed_documents = document.lower().split()  # Basic tokenization and lowercasing
            for word in processed_documents:
                if doc_id not in self.inverted_index[word]:
                    self.inverted_index[word].append(doc_id)
        # Save the inverted index
        self.storage_manager.save_inverted_index(self.inverted_index)

    def load_data(self):
        # Load the inverted index, document vectors, and vectorizer from files
        self.inverted_index = self.storage_manager.load_inverted_index()
        self.vectorizer = self.storage_manager.load_vectorizer()
        self.document_vectors = self.storage_manager.load_document_vectors()

    def search(self, query):
        query_words = query.split()
        if not query_words:
            return []

        # Initialize result set with documents containing the first query word
        result_set = set(self.inverted_index.get(query_words[0], []))

        # Intersect results with other query words
        for word in query_words[1:]:
            if word in self.inverted_index:
                result_set.intersection_update(self.inverted_index[word])
            else:
                return []

        return list(result_set)

    def search_vectors(self, query_vector):
        # Calculate cosine similarity between query vector and document vectors
        similarity_scores = cosine_similarity(query_vector, self.document_vectors)

        # Convert similarity scores to a flat list
        similarity_scores = similarity_scores.flatten()

        # Return similarity scores
        return similarity_scores
