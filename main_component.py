from evaluate.evaluate import Evaluate
from interface.cli_interface import CLIInterface
# import nltk
# from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer


def filter_documents_by_ids(documents, ids):
    return [doc for doc in documents if doc['_id'] in ids]


# /home/abdallh/Documents/webis-touche2020/
# remember to give the title priority over the text in the search
# tune the model to this need
def main():
    interface = CLIInterface()
    # interface.load_data('/home/abdallh/Documents/webis-touche2020/corpus.jsonl')
    qrels = interface.load_qrels('/home/abdallh/Documents/webis-touche2020/qrels/test.tsv')
    interface.load_queries('/home/abdallh/Documents/webis-touche2020/queries.jsonl')
    # interface.run()
    # Example of Searching and Evaluating
    query_id = '1'
    query_text = interface.queries[query_id]
    with open('/home/abdallh/Documents/webis-touche2020/qrels/test.tsv', 'r') as f:
        qrels = f.readlines()

    relevant_docs = [line.split()[1] for line in qrels if line.split()[0] == query_id]
    # retrieved_results = interface.search_query_with_id(query_text)
    retrieved_ids, retrieved_results = interface.run_final(query_text)
    retrieved_docs_id = retrieved_results
    matching_documents = interface.get_docs_by_ids('/home/abdallh/Documents/webis-touche2020/corpus.jsonl',
                                                   retrieved_ids)
    for rank, (doc_id, score) in enumerate(retrieved_results[:10]):
        if doc_id in matching_documents:
            document = matching_documents[doc_id]
            print(f"Rank {rank + 1}: Document {doc_id}, Similarity Score: {score}")
            interface.print_ranked_data(document[doc_id])

        # self.print_ranked_data(self.documents[doc_id])

        # print(self.documents[doc_id])
        print()

    print("this query", query_text)
    # Extract the document IDs from retrieved_results
    # retrieved_ids = [doc_id for doc_id, score in retrieved_results[:500]]

    # Filter the documents
    # filtered_documents = filter_documents_by_ids(interface.documents, retrieved_ids)
    # retrieved_docs = [
    #     {
    #         '_id': doc['_id'],
    #         'title': doc['title'],
    #         'text': doc['text'],
    #         'score': next(score for id, score in retrieved_results if id == doc['_id'])
    #     }
    #     for doc in filtered_documents
    # ]
    # # Create the retrieved_docs_id list
    # retrieved_docs_id = [doc['_id'] for doc in filtered_documents]
    print(len(retrieved_docs_id))
    print(len(relevant_docs))
    e = Evaluate(actual=relevant_docs, predicted=retrieved_docs_id, k=10)
    e.print_all()
    # print(len(relevant_docs))
    # print(len(retrieved_docs_id))

    # precision, recall = interface.evaluate(query_id, retrieved_docs, qrels)
    #
    # print(f"Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    main()
