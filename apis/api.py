from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = Flask(__name__)

# Initialize variables
documents = []
document_vectors = None
vectorizer = TfidfVectorizer()


# Load and preprocess documents
def load_documents(file_path):
    global documents, document_vectors
    with open(file_path, 'r') as file:
        documents = [json.loads(line) for line in file]
    preprocessed_documents = [doc['text'] for doc in documents]
    document_vectors = vectorizer.fit_transform(preprocessed_documents)


# Search for a query
def search(query_text):
    query_vector = vectorizer.transform([query_text])
    similarity_scores = cosine_similarity(query_vector, document_vectors).flatten()
    ranked_doc_indices = similarity_scores.argsort()[::-1]
    ranked_docs = [(documents[i]['_id'], similarity_scores[i]) for i in ranked_doc_indices]
    return ranked_docs


# Search for a query
def search(query_text, selected_doc_ids):
    query_vector = vectorizer.transform([query_text])
    if selected_doc_ids:
        selected_indices = [i for i, doc in enumerate(documents) if doc['_id'] in selected_doc_ids]
        document_vectors_subset = document_vectors[selected_indices]
        similarity_scores = cosine_similarity(query_vector, document_vectors_subset).flatten()
        ranked_doc_indices = similarity_scores.argsort()[::-1]
        ranked_docs = [(documents[selected_indices[i]]['_id'], similarity_scores[i]) for i in ranked_doc_indices]
    else:
        similarity_scores = cosine_similarity(query_vector, document_vectors).flatten()
        ranked_doc_indices = similarity_scores.argsort()[::-1]
        ranked_docs = [(documents[i]['_id'], similarity_scores[i]) for i in ranked_doc_indices]
    return ranked_docs


# API endpoint for searching
# POST /search
# {
#     "query": "Should teachers get tenure?"
# }
# [
#     {
#         "_id": "c67482ba-2019-04-18T13:32:05Z-00000-000",
#         "score": 0.5
#     },
#     {
#         "_id": "other_document_id",
#         "score": 0.3
#     },
#     ...
# ]


# API endpoint to get all documents
@app.route('/documents', methods=['GET'])
def get_documents():
    return jsonify(documents)


# API endpoint for searching
# POST /search
# {
#     "query": "Should teachers get tenure?",
#     "selected_doc_ids": ["c67482ba-2019-04-18T13:32:05Z-00000-000", "other_document_id"]
# }

# [
#     {
#         "_id": "c67482ba-2019-04-18T13:32:05Z-00000-000",
#         "score": 0.5
#     },
#     {
#         "_id": "other_document_id",
#         "score": 0.3
#     },
#     ...
# ]

@app.route('/search', methods=['POST'])
def search_endpoint():
    query_text = request.json['query']
    selected_doc_ids = request.json.get('selected_doc_ids', [])
    ranked_docs = search(query_text, selected_doc_ids)
    return jsonify(ranked_docs)


@app.route('/search', methods=['POST'])
def search_endpoint():
    query_text = request.json['query']
    ranked_docs = search(query_text)
    return jsonify(ranked_docs)


# API endpoint to get document by ID
@app.route('/document/<doc_id>', methods=['GET'])
def get_document(doc_id):
    document = next((doc for doc in documents if doc['_id'] == doc_id), None)
    if document:
        return jsonify(document)
    else:
        return jsonify({'error': 'Document not found'}), 404


# API endpoint to add a new document
@app.route('/document', methods=['POST'])
def add_document():
    new_document = request.json
    documents.append(new_document)
    document_vectors = vectorizer.fit_transform([doc['text'] for doc in documents])
    return jsonify({'message': 'Document added successfully'})


# API endpoint to delete a document by ID
@app.route('/document/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    global documents, document_vectors
    documents = [doc for doc in documents if doc['_id'] != doc_id]
    document_vectors = vectorizer.fit_transform([doc['text'] for doc in documents])
    return jsonify({'message': 'Document deleted successfully'})


# API endpoint to update a document by ID
@app.route('/document/<doc_id>', methods=['PUT'])
def update_document(doc_id):
    document_data = request.json
    for doc in documents:
        if doc['_id'] == doc_id:
            doc.update(document_data)
            break
    document_vectors = vectorizer.fit_transform([doc['text'] for doc in documents])
    return jsonify({'message': 'Document updated successfully'})


# API endpoint to evaluate the IR system
@app.route('/evaluate', methods=['POST'])
def evaluate_system():
    query_id = request.json['query_id']
    retrieved_docs = request.json['retrieved_docs']
    qrels = request.json['qrels']
    # Implement evaluation logic here
    return jsonify({'message': 'Evaluation completed'})


# API endpoint to reindex documents
@app.route('/reindex', methods=['POST'])
def reindex_documents():
    load_documents('corpus.jsonl')
    return jsonify({'message': 'Documents reindexed successfully'})


# API endpoint to clear index
@app.route('/clear-index', methods=['POST'])
def clear_index():
    global documents, document_vectors
    documents = []
    document_vectors = None
    return jsonify({'message': 'Index cleared successfully'})


# Main function to start the server
if __name__ == '__main__':
    load_documents('corpus.jsonl')
    app.run(debug=True)
