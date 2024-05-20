from sklearn.metrics.pairwise import cosine_similarity

from query.query_processor import QueryProcessor
from indexing.indexer import Indexer
from ranking.ranker import Ranker
import xml.etree.ElementTree as ET
import json
import pandas as pd


class CLIInterface:
    def __init__(self):
        self.queries = {}
        self.query_processor = QueryProcessor()
        self.indexer = Indexer()
        self.ranker = Ranker()
        self.documents = []

    def load_queries(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                query = json.loads(line)
                self.queries[query['_id']] = query['text']

    def load_qrels(self, file_path):
        qrels = pd.read_csv(file_path, sep='\t', names=["query_id", "corpus_id", "score"])
        return qrels

    def load_data(self, file_path):
        # Load the dataset
        # self.documents = self.load_dataset(file_path)
        self.documents = self.read_jsonl_file(file_path)
        # self.process_documents(self.documents)

    def evaluate(self, query_id, retrieved_docs, qrels):
        rel = ['e4cd6b8-2019-04-18T15:15:19Z-00004-000', 'e4cd6b8-2019-04-18T15:15:19Z-00001-000',
               'e4cd6b8-2019-04-18T15:15:19Z-00005-000', 'e9b44971-2019-04-18T13:56:01Z-00003-000',
               '758ea5f9-2019-04-18T16:05:18Z-00004-000', '81c7fb51-2019-04-18T19:58:34Z-00002-000',
               '8294b441-2019-04-18T17:22:30Z-00003-000', '8440ef2-2019-04-18T12:02:18Z-00000-000',
               'de7919e0-2019-04-18T16:05:16Z-00001-000', '3466ccde-2019-04-18T15:56:31Z-00006-000',
               '4381b332-2019-04-19T12:47:35Z-00017-000', '45462ad0-2019-04-18T13:17:49Z-00000-000',
               '55e10797-2019-04-18T15:21:42Z-00006-000', '5b6b2f9-2019-04-18T15:28:19Z-00000-000',
               'd0e5c093-2019-04-18T18:40:47Z-00005-000', '5dce2de2-2019-04-18T15:41:55Z-00001-000',
               'a5ca39dc-2019-04-18T11:43:36Z-00005-000', 'b1a6f17a-2019-04-18T15:54:21Z-00002-000',
               'b1a6f17a-2019-04-18T15:54:21Z-00001-000', 'b4849efd-2019-04-18T17:56:23Z-00002-000',
               '117d4c1a-2019-04-18T18:37:03Z-00004-000', 'c80f9596-2019-04-18T15:38:23Z-00003-000',
               'c958dc5a-2019-04-18T17:51:05Z-00001-000', '3307f209-2019-04-18T15:40:14Z-00004-000']
        relevant_docs = qrels[qrels['query_id'] == query_id]
        print(relevant_docs)
        print(relevant_docs[pd.to_numeric(relevant_docs['score'], errors='coerce') > 0])
        # print(int(relevant_docs['score']))
        # print(int(relevant_docs['score']) > 0, relevant_docs['score']['corpus_id'])
        relevant_doc_ids = relevant_docs[pd.to_numeric(relevant_docs['score'], errors='coerce') > 0][
            'corpus_id'].tolist()
        retrieved_doc_ids = [doc_id for doc_id, _ in retrieved_docs]
        print("this the reirived", retrieved_doc_ids)
        print("this the relevant", relevant_doc_ids)
        # for rank, doc_id in enumerate(retrieved_doc_ids):
        #     # doc_id = int(doc_id)
        #     # print(f"Rank {rank + 1}: Document {doc_id + 1}: {documents[doc_id]}")
        #     # print(self.documents[0])
        #     self.print_ranked_data(self.documents[doc_id])

        tp = len(set(relevant_doc_ids).intersection(retrieved_doc_ids))
        print("this the tp:", tp)
        precision = tp / len(retrieved_doc_ids) if retrieved_doc_ids else 0
        recall = tp / len(relevant_doc_ids) if relevant_doc_ids else 0

        return precision, recall

    def search(self, query_text):
        self.indexer.index_documents(self.documents)

        # processed_query = self.query_processor.process_query(query)
        processed_query = self.query_processor.complete_process_query(query_text)
        # Transform the processed query to VSM using the same vectorizer as the documents
        query_vector = self.indexer.vectorizer.transform(processed_query)

        # Perform the search to get similarity scores
        similarity_scores = self.indexer.search_vectors(query_vector)
        # Rank the results based on similarity scores
        ranked_results = self.ranker.rank_vectors_results(similarity_scores)
        print(similarity_scores)
        print(ranked_results)
        return [(self.documents[i]['_id'], similarity_scores[i]) for i in ranked_results]
        # return [(self.documents[doc_id]['_id'], similarity_scores[doc_id]) for rank, doc_id in
        #         enumerate(ranked_results)]

    def search2(self, query_text):
        self.indexer.index_documents(self.documents)
        # self.indexer.load_data()
        query_text.lower()
        query_vector = self.indexer.vectorizer.transform([query_text])
        similarity_scores = cosine_similarity(query_vector, self.indexer.document_vectors).flatten()
        ranked_doc_indices = similarity_scores.argsort()[::-1]
        print(similarity_scores)
        print(ranked_doc_indices)
        for rank, doc_id in enumerate(ranked_doc_indices):
            # doc_id = int(doc_id)
            # print(f"Rank {rank + 1}: Document {doc_id + 1}: {documents[doc_id]}")
            # print(self.documents[0])
            self.print_ranked_data(self.documents[doc_id])

        return [(self.documents[i]['_id'], similarity_scores[i]) for i in ranked_doc_indices]

    def run(self):
        print("Welcome to the Information Retrieval System!")

        # Sample documents for demonstration purposes
        documents = [
            "Apple is a fruit.",
            "Banana is also a fruit.",
            "Both apple and banana are healthy.",
            "Orange is another type of fruit."
        ]

        # Index the sample documents
        # self.indexer.index_documents(documents)
        # self.indexer.index_documents(self.documents)
        self.indexer.load_data()
        while True:
            query = input("\nEnter your search query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            # Process the query
            # processed_query = self.query_processor.process_query(query)
            processed_query = self.query_processor.complete_process_query(query)
            # Join the list of strings into a single string
            joined_string = " ".join(processed_query)
            print(joined_string)
            # Transform the processed query to VSM using the same vectorizer as the documents
            query_vector = self.indexer.vectorizer.transform([joined_string])

            # Perform the search to get similarity scores
            similarity_scores = self.indexer.search_vectors(query_vector)

            # Rank the results based on similarity scores
            ranked_results = self.ranker.rank_vectors_results(similarity_scores)
            # processed_query = self.query_processor.process_query(query)
            # results = self.indexer.search(processed_query)
            # Display the ranked search results
            # Check the top similarity score against the threshold
            # top_similarity_score = similarity_scores[0] if similarity_scores.size>0 else 0
            print("this the similarity score:", similarity_scores)
            print(ranked_results)
            # if top_similarity_score ==0:
            #     # If the top score is below the threshold, return an unsure message
            #     print("The system is unsure about the query. No relevant documents found.")
            #     # return "The system is unsure about the query. No relevant documents found."
            # else:
            print("\nSearch Results (ranked by relevance):")
            for rank, doc_id in enumerate(ranked_results[:10]):
                # doc_id = int(doc_id)
                # print(f"Rank {rank + 1}: Document {doc_id + 1}: {documents[doc_id]}")
                # print(self.documents[0])
                self.print_ranked_data(self.documents[doc_id])
                # self.process_documents(self.documents[doc_id])
                # if results:
                #     # ranked_results = self.ranker.rank_results(results)
                #     print("\nSearch Results:")
                #     for doc_id in ranked_results:
                #         print(f"Document {doc_id + 1}: {documents[doc_id]}")
                #
                # else:
                #     print("No results found.")

            ranked_results = self.ranker.rank_vectors_results_reutrn_tuples(similarity_scores)

            # Print top-k ranked documents
            top_k = 10
            for rank, (doc_id, score) in enumerate(ranked_results[:top_k]):
                print(f"Rank {rank + 1}: Document {doc_id + 1}, Similarity Score: {score}")
                self.print_ranked_data(self.documents[doc_id])

                # print(self.documents[doc_id])
                print()

            return ranked_results

    def load_dataset(self, file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Extract documents
        documents = []
        for doc in root.findall('document'):
            doc_id = doc.get('id')
            text = doc.find('text').text
            # Add other metadata as needed
            documents.append((doc_id, text))

        return documents

    def read_jsonl_file(self, file_path):
        documents = []

        # Open the JSON Lines file
        with open(file_path, 'r', encoding='utf-8') as file:
            x = 0

            # Read each line from the file
            for line in file:
                x += 1
                # if x > 100:
                #     break
                print(x)
                # Parse the line as a JSON object
                doc = json.loads(line)
                # Append the document to the list of documents
                documents.append(doc)

        return documents

    def process_documents(self, documents):
        # Process each document as needed for your IR system
        for doc in documents:
            # Extract document fields

            doc_id = doc.get('_id')
            doc_title = doc.get('title')
            doc_text = doc.get('text')
            doc_metadata = doc.get('metadata')

            # Process document data as required by your IR system
            # For example, you might index the document text, title, and metadata

            # Print the document information as a demonstration
            print(f"Document ID: {doc_id}")
            print(f"Title: {doc_title}")
            print(f"Text: {doc_text[:100]}...")  # Print the first 100 characters of the text
            print(f"Metadata: {doc_metadata}")
            print()

    def print_ranked_data(self, doc):
        # Process each document as needed for your IR system
        # Extract document fields

        doc_id = doc.get('_id')
        doc_title = doc.get('title')
        doc_text = doc.get('text')
        doc_metadata = doc.get('metadata')

        # Process document data as required by your IR system
        # For example, you might index the document text, title, and metadata

        # Print the document information as a demonstration
        print(f"Document ID: {doc_id}")
        print(f"Title: {doc_title}")
        print(f"Text: {doc_text[:100]}...")  # Print the first 100 characters of the text
        print(f"Metadata: {doc_metadata}")
        print()
