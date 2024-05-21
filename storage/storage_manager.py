import os
import pickle

import numpy as np


class StorageManager:
    def __init__(self, base_path='data/', storage_dir='data'):
        self.batch_size = 1000
        self.base_path = base_path
        self.checkpoint_file = 'file_checkpoint.pkl'
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.vectorizer_file = os.path.join(self.storage_dir, 'vectorizer.pkl')
        self.document_vectors_dir = os.path.join(self.storage_dir, 'document_vectors')
        os.makedirs(self.document_vectors_dir, exist_ok=True)
        self.inverted_index_file = os.path.join(self.storage_dir, 'inverted_index.pkl')
        self.vocabulary_file = os.path.join(self.storage_dir, 'vocabulary.pkl')

    def save_batch_document_vectors(self, vectors, batch_id):
        batch_file = os.path.join(self.document_vectors_dir, f'document_vectors_{batch_id}.pkl')
        with open(batch_file, 'wb') as f:
            pickle.dump(vectors, f)

    def load_all_document_vectors(self):
        all_vectors = []
        max_length = 0
        for batch_file in sorted(os.listdir(self.document_vectors_dir)):
            with open(os.path.join(self.document_vectors_dir, batch_file), 'rb') as f:
                batch_vectors = pickle.load(f)
                max_length = max(max_length, batch_vectors.shape[1])
                all_vectors.append(batch_vectors)
        # Pad shorter vectors with zeros to match the length of the longest vector
        padded_vectors = []
        for batch_vectors in all_vectors:
            num_docs, vec_length = batch_vectors.shape
            if vec_length < max_length:
                padding = np.zeros((num_docs, max_length - vec_length))
                batch_vectors = np.hstack((batch_vectors, padding))
            padded_vectors.append(batch_vectors)

        # Concatenate all batch vectors into a single array
        document_vectors = np.vstack(padded_vectors)
        return document_vectors  # return all_vectors

    # def save_batch_document_vectors(self, batch_vectors, batch_start):
    #     np.save(f'document_vectors_{batch_start}.npy', batch_vectors.toarray())

    def save_checkpoint(self, checkpoint):
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)

    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None

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
    def batch_file_exists(self, batch_id):
        batch_file = os.path.join(self.document_vectors_dir, f'document_vectors_{batch_id}.pkl')
        return os.path.exists(batch_file)
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

    def get_num_batches(self):
        return len(os.listdir(self.document_vectors_dir))

    def load_batch_document_vectors(self, batch_id):
        batch_file = os.path.join(self.document_vectors_dir, f'document_vectors_{batch_id}.pkl')
        if not os.path.exists(batch_file):

            raise FileNotFoundError(f"Batch file {batch_file} does not exist")

        with open(batch_file, 'rb') as f:
            document_vectors = pickle.load(f)
        return document_vectors

    def reset_project(self):
        # Step 1: Delete existing files and directories
        files_to_delete = ['document_vectors', 'inverted_index', 'vocabulary', 'checkpoints']
        for file_name in files_to_delete:
            if os.path.exists(file_name):
                if os.path.isfile(file_name):
                    os.remove(file_name)
                elif os.path.isdir(file_name):
                    os.rmdir(file_name)

        # Step 2: Reinitialize data structures
        inverted_index = {}
        vocabulary = set()
        # Other data structure reinitialization...

        # Step 3: Run initialization code
        # Run code to recreate directories, load initial data, etc.

        # Step 4: Verify reset
        # Check if files and data structures are empty or contain default values

        print("Project reset complete.")
