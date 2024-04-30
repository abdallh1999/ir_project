from query.query_processor import QueryProcessor
from indexing.indexer import Indexer
from ranking.ranker import Ranker


class CLIInterface:
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.indexer = Indexer()
        self.ranker = Ranker()

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
        self.indexer.index_documents(documents)

        while True:
            query = input("\nEnter your search query (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            # Process the query
            # processed_query = self.query_processor.process_query(query)
            processed_query = self.query_processor.complete_process_query(query)

            # Transform the processed query to VSM using the same vectorizer as the documents
            query_vector = self.indexer.vectorizer.transform(processed_query)

            # Perform the search to get similarity scores
            similarity_scores = self.indexer.search_vectors(query_vector)

            # Rank the results based on similarity scores
            ranked_results = self.ranker.rank_vectors_results(similarity_scores)
            # processed_query = self.query_processor.process_query(query)
            # results = self.indexer.search(processed_query)
            # Display the ranked search results
            # Check the top similarity score against the threshold
            top_similarity_score = similarity_scores[0] if similarity_scores.size>0 else 0

            if top_similarity_score < 0.05:
                # If the top score is below the threshold, return an unsure message
                print("The system is unsure about the query. No relevant documents found.")
                # return "The system is unsure about the query. No relevant documents found."
            else:
                print("\nSearch Results (ranked by relevance):")
                for rank, doc_id in enumerate(ranked_results):
                    # doc_id = int(doc_id)
                    print(f"Rank {rank + 1}: Document {doc_id + 1}: {documents[doc_id]}")
                # if results:
                #     # ranked_results = self.ranker.rank_results(results)
                #     print("\nSearch Results:")
                #     for doc_id in ranked_results:
                #         print(f"Document {doc_id + 1}: {documents[doc_id]}")
                #
                # else:
                #     print("No results found.")
