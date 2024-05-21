from interface.cli_interface import CLIInterface
# import nltk
# from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer


# /home/abdallh/Documents/webis-touche2020/
# remember to give the title priority over the text in the search
# tune the model to this need
def main():
    interface = CLIInterface()
    interface.load_data('/home/abdallh/Documents/webis-touche2020/corpus.jsonl')
    qrels = interface.load_qrels('/home/abdallh/Documents/webis-touche2020/qrels/test.tsv')
    interface.load_queries('/home/abdallh/Documents/webis-touche2020/queries.jsonl')
    # interface.run()
    # Example of Searching and Evaluating
    query_id = '35'
    query_text = interface.queries[query_id]
    retrieved_docs = interface.search_query(query_text)
    precision, recall = interface.evaluate(query_id, retrieved_docs, qrels)

    print(f"Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    main()

