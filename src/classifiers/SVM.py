from sklearn.svm import SVC
from src.classifiers.Classifier import Classifier


class SVM(Classifier):

    def __init__(self, c=1.0, kernel="linear"):
        super().__init__()
        self.svm = SVC(c, kernel)

    def fit(self, x, y):
        self.svm.fit(x, y)

    def predict(self, x):
        return self.svm.predict(x)
