from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate(y_true, pred):
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1 Score": f1_score(y_true, pred, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_true, pred),
    }
