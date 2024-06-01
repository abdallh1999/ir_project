# from interface.cli_interface import CLIInterface
# import nltk
# from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer

from evaluate.evaluate import Evaluate
from interface.cli_interface import CLIInterface

# /home/abdallh/Documents/webis-touche2020/
# remember to give the title priority over the text in the search
# tune the model to this need
# def main():
#     interface = CLIInterface()
#     interface.load_data('/home/abdallh/Documents/webis-touche2020/corpus.jsonl')
#     qrels = interface.load_qrels('/home/abdallh/Documents/webis-touche2020/qrels/test.tsv')
#     interface.load_queries('/home/abdallh/Documents/webis-touche2020/queries.jsonl')
#     interface.search_query_with_id('1')
# Example of Searching and Evaluating
# query_id = '35'
# query_text = interface.queries[query_id]
# retrieved_docs = interface.search2(query_text)
# precision, recall = interface.evaluate(query_id, retrieved_docs, qrels)
#
# print(f"Precision: {precision}, Recall: {recall}")
# import ir_datasets
# import os
#
# # Load the dataset
# dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")
#
# # Step 2: Iterate through queries and store them
# queries = {query.query_id: query.text for query in dataset.queries_iter()}
# print("Queries loaded:", queries)
#
# # Step 3: Extract relevant document IDs from qrels
# qrels = {}
# for qrel in dataset.qrels_iter():
#     if qrel.query_id not in qrels:
#         qrels[qrel.query_id] = []
#     qrels[qrel.query_id].append(qrel.doc_id)
#
# print("Qrels loaded:", qrels)
#
# # Step 4: Retrieve and read only relevant documents
# # Set to store unique relevant document IDs
# relevant_doc_ids = set()
# for doc_ids in qrels.values():
#     relevant_doc_ids.update(doc_ids)
#
# # Dictionary to store the relevant documents
# relevant_docs = {query_id: [] for query_id in queries.keys()}
#
# # Iterate through the dataset and collect relevant documents
# for doc in dataset.docs_iter():
#     if doc.doc_id in relevant_doc_ids:
#         for query_id, doc_ids in qrels.items():
#             if doc.doc_id in doc_ids:
#                 relevant_docs[query_id].append(doc)
#
# # Step 5: Write relevant documents to files
# # Create output directory if it doesn't exist
# output_dir = "relevant_docs"
# os.makedirs(output_dir, exist_ok=True)
#
# for query_id, docs in relevant_docs.items():
#     query_output_path = os.path.join(output_dir, f"query_{query_id}.txt")
#     with open(query_output_path, 'w', encoding='utf-8') as file:
#         file.write(f"Query ID: {query_id}\nQuery: {queries[query_id]}\n\n")
#         for doc in docs:
#             file.write(f"Document ID: {doc.doc_id}\n")
#             file.write(f"Title: {doc.title}\n")
#             file.write(f"Condition: {doc.condition}\n")
#             file.write(f"Summary: {doc.summary}\n")
#             file.write(f"Detailed Description: {doc.detailed_description}\n")
#             file.write(f"Eligibility: {doc.eligibility}\n")
#             file.write("\n" + "="*80 + "\n\n")
#
# print("Relevant documents saved.")

# # Step 5: Write relevant documents to files
# # Create output directory if it doesn't exist
# output_dir = "relevant_docs"
# os.makedirs(output_dir, exist_ok=True)
#
# for query_id, docs in relevant_docs.items():
#     query_output_dir = os.path.join(output_dir, f"query_{query_id}")
#     os.makedirs(query_output_dir, exist_ok=True)
#     for doc in docs:
#         doc_path = os.path.join(query_output_dir, f"{doc.doc_id}.txt")
#         with open(doc_path, 'w', encoding='utf-8') as file:
#             file.write(doc.title)
#
# print("Relevant documents saved.")

# import ir_datasets
#
# if __name__ == '__main__':
#     # main()
#
#     dataset = ir_datasets.load("clinicaltrials/2021/trec-ct-2021")
#     for doc in dataset.docs_iter():
#         print(doc)
#     for query in dataset.queries_iter():
#         query  # namedtuple<query_id, text>
#         print(query)
#     for qrel in dataset.qrels_iter():
#         qrel  # namedtuple<query_id, doc_id, relevance, iteration>
#         print(qrel)
# # Sample data
# documents = [
#     "Apple is a fruit.",
#     "Banana is also a fruit.",
#     "Both apple and banana are healthy.",
#     "Orange is another type of fruit."
# ]
#
# query = "I like apple and banana."
#
# # Create a TF-IDF vectorizer
# vectorizer = TfidfVectorizer()
#
# # Fit the vectorizer to the documents and transform the documents to VSM
# document_vectors = vectorizer.fit_transform(documents)
#
# # Transform the query to VSM using the same vectorizer
# query_vector = vectorizer.transform([query])
#
# # Display the vocabulary (terms in the vector space)
# vocabulary = vectorizer.get_feature_names_out()
# print("Vocabulary (terms):", vocabulary)
#
# # Display the TF-IDF values for each document
# print("\nTF-IDF values for each document:")
# for doc_id, doc_vector in enumerate(document_vectors):
#     print(f"Document {doc_id + 1}:")
#     tfidf_values = doc_vector.toarray().flatten()  # Convert the sparse matrix to a 1D array
#     print(dict(zip(vocabulary, tfidf_values)))
#
# # Display the TF-IDF values for the query
# print("\nTF-IDF values for the query:")
# query_tfidf_values = query_vector.toarray().flatten()  # Convert the sparse matrix to a 1D array
# print(dict(zip(vocabulary, query_tfidf_values)))

# nltk.download()

interface = CLIInterface()
retrieved_ids, retrieved_results ,relevant_docs = interface.run_final_dataset2(
    'Patient is a 45-year-old man with a history of anaplastic astrocytoma of the spine complicated by severe lower extremity weakness and urinary retention s/p Foley catheter, high-dose steroids, hypertension, and chronic pain. The tumor is located in the T-L spine, unresectable anaplastic astrocytoma s/p radiation. Complicated by progressive lower extremity weakness and urinary retention. Patient initially presented with RLE weakness where his right knee gave out with difficulty walking and right anterior thigh numbness. MRI showed a spinal cord conus mass which was biopsied and found to be anaplastic astrocytoma. Therapy included field radiation t10-l1 followed by 11 cycles of temozolomide 7 days on and 7 days off. This was followed by CPT-11 Weekly x4 with Avastin Q2 weeks/ 2 weeks rest and repeat cycle. ')
# print(retrieved_ids)
# print(retrieved_results)
matching_documents = interface.get_docs_by_ids_dataset2('clinicaltrials/2021/trec-ct-2021'
                                               , retrieved_ids)
e = Evaluate(actual=relevant_docs["1"], predicted=retrieved_ids, k=10)
precision, recall, f1 = e.calculate_metrics()
e.print_all()
# print(retrieved_ids)
# print(relevant_docs["1"])
for rank, (doc_id, score) in enumerate(retrieved_results[:10]):
    document = matching_documents.get(doc_id)
    if document:
        print(f"Rank {rank + 1}: Document {doc_id}, Similarity Score: {score}")
        interface.print_ranked_dataset_2(document)
    else:
        print(f"Rank {rank + 1}: Document {doc_id} not found")
