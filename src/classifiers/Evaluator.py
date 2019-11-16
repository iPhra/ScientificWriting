from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


def evaluate_classifier(ytrue, ypredicted):
    acc = accuracy_score(ytrue, ypredicted)
    print("Accuracy: ", acc)
    print("Precision: ", precision_score(ytrue, ypredicted))
    print("Recall: ", recall_score(ytrue, ypredicted))
    print("F1 score: ", f1_score(ytrue, ypredicted))
    print("ROC-AUC score: ", roc_auc_score(ytrue, ypredicted))
    print("Confusion matrix:\n", confusion_matrix(ytrue, ypredicted))
    return acc
