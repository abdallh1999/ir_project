import numpy as np
class Ranker:
    def rank_results(self, results):
        # Basic ranking based on document frequency
        # You can implement more sophisticated ranking algorithms
        # For this example, we'll return results in the order of appearance
        return results

    def rank_vectors_results(self, similarity_scores):
        # Convert similarity scores to an array and sort the indices in descending order
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # Return sorted indices, which indicate document IDs in descending order of relevance
        return sorted_indices

    def rank_vectors_results_reutrn_tuples(self, similarity_scores):
        # Convert similarity scores to an array and sort the indices in descending order
        sorted_indices = np.argsort(similarity_scores)[::-1]

        # Create a list of tuples (doc_id, score)
        ranked_results = [(doc_id, similarity_scores[doc_id]) for doc_id in sorted_indices]

        # Return the list of tuples
        return ranked_results
