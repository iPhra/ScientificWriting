from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classifier(ytrue, ypredicted):
    print(classification_report(ytrue, ypredicted))
    print(confusion_matrix(ytrue, ypredicted))
