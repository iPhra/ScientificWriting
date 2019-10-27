from sklearn.naive_bayes import MultinomialNB
from src.classifiers.Classifier import Classifier


class NaiveBayes(Classifier):

    def __init__(self, alpha):
        self.nb = MultinomialNB(alpha)

    def fit(self, x, y):
        self.nb.fit(x, y)

    def predict(self, x):
        return self.nb.predict(x)
