from sklearn.ensemble import RandomForestClassifier
from src.classifiers.Classifier import Classifier


class RandomForest(Classifier):

    def __init__(self, n=300):
        super().__init__()
        self.rf = RandomForestClassifier(n_estimators=n)

    def fit(self, x, y):
        self.rf.fit(x, y)

    def predict(self, x):
        return self.rf.predict(x)
