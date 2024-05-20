# 1. Precision@10: Proportion of relevant documents among the top 10 retrieved documents.
# 2. Recall: Proportion of relevant documents that were retrieved out of the total number of relevant documents.
# 3. MAP (Mean Average Precision): Average of precision values at different recall levels.
# 4. Precision: Proportion of retrieved documents that are relevant.
# 5. MRR (Mean Reciprocal Rank): Average of the reciprocal ranks of the first relevant document found.
# 6. Rank Reciprocal Mean: Average reciprocal rank for all queries.
def precision_at_k(actual, predicted, k):
    predicted = predicted[:k]
    tp = len(set(predicted) & set(actual))
    return tp / k

def recall(actual, predicted):
    tp = len(set(predicted) & set(actual))
    return tp / len(actual)

def average_precision_at_k(actual, predicted, k):
    precisions = [precision_at_k(actual, predicted, i+1) for i in range(k) if predicted[i] in actual]
    if not precisions:
        return 0
    return sum(precisions) / min(k, len(actual))

def mean_reciprocal_rank(actual, predicted):
    for i, p in enumerate(predicted):
        if p in actual:
            return 1 / (i + 1)
    return 0

# Example usage
actual = [1, 2, 3]
predicted = [3, 1, 5, 2, 7]
k = 3

print("Precision@{}: {}".format(k, precision_at_k(actual, predicted, k)))
print("Recall: {}".format(recall(actual,predicted)))
print("MAP: {}".format(average_precision_at_k(actual,predicted,k)))
print("MRR: {}".format(mean_reciprocal_rank(actual,predicted)))





from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
from your_ir_system import InformationRetrievalSystem  # Import your IR system

# Assuming you have your dataset in a list of documents and their corresponding relevance labels
documents = [...]
relevance_labels = [...]

# Split the data into training, validation, and test sets
train_docs, temp_docs, train_labels, temp_labels = train_test_split(documents, relevance_labels, test_size=0.4)
validation_docs, test_docs, validation_labels, test_labels = train_test_split(temp_docs, temp_labels, test_size=0.5)

# Initialize your IR system and train it on the training set
ir_system = InformationRetrievalSystem()
ir_system.train(train_docs, train_labels)

# Evaluate the IR system on the validation set
validation_results = ir_system.search(validation_docs)
precision = precision_score(validation_labels, validation_results)
recall = recall_score(validation_labels, validation_results)
f1 = f1_score(validation_labels, validation_results)
ndcg = ndcg_score(validation_labels, validation_results)

print(f'Precision: {precision}, Recall: {recall}, F1-Score: {f1}, NDCG: {ndcg}')

# Adjust model parameters based on validation results and iterate as needed
# ...

# After tuning, evaluate on the test set
test_results = ir_system.search(test_docs)
test_precision = precision_score(test_labels, test_results)
test_recall = recall_score(test_labels, test_results)
test_f1 = f1_score(test_labels, test_results)
test_ndcg = ndcg_score(test_labels, test_results)

print(f'Test Precision: {test_precision}, Test Recall: {test_recall}, Test F1-Score: {test_f1}, Test NDCG: {test_ndcg}')
