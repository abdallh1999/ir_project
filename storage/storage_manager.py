import pickle


class StorageManager:
    def __init__(self, base_path='data/'):
        self.base_path = base_path

    def save_inverted_index(self, inverted_index, filename='inverted_index.pkl'):
        file_path = self.base_path + filename
        with open(file_path, 'wb') as file:
            pickle.dump(inverted_index, file)

    def load_inverted_index(self, filename='inverted_index.pkl'):
        file_path = self.base_path + filename
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def save_document_vectors(self, document_vectors, filename='document_vectors.pkl'):
        file_path = self.base_path + filename
        with open(file_path, 'wb') as file:
            pickle.dump(document_vectors, file)

    def load_document_vectors(self, filename='document_vectors.pkl'):
        file_path = self.base_path + filename
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def save_vectorizer(self, vectorizer, filename='tfidf_vectorizer.pkl'):
        file_path = self.base_path + filename
        with open(file_path, 'wb') as file:
            pickle.dump(vectorizer, file)

    def load_vectorizer(self, filename='tfidf_vectorizer.pkl'):
        file_path = self.base_path + filename
        with open(file_path, 'rb') as file:
            return pickle.load(file)

    def save_vocabulary(self, vocabulary, filename='vocabulary.txt'):
        file_path = self.base_path + filename
        with open(file_path, 'w') as file:
            for term in vocabulary:
                file.write(f"{term}\n")

    def load_vocabulary(self, filename='vocabulary.txt'):
        file_path = self.base_path + filename
        vocabulary = []
        with open(file_path, 'r') as file:
            vocabulary = [line.strip() for line in file]
        return vocabulary
