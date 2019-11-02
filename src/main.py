import pandas as pd
from src import preprocess
from src.features.features import Features
from src.classifiers import NaiveBayes, Classifier, SVM, RandomForest, Evaluator
from sklearn.model_selection import train_test_split


class Main:

    def __init__(self, parameters):
        self.parameters = parameters
        self.df = None
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
        self.load_dataset()


    def load_dataset(self):
        self.df = pd.read_csv("../datasets/preprocessed/" + self.parameters["dataset"],
                              encoding="ISO-8859-1", names=["sentiment", "tweet"])


    def preprocess(self):
        preprocess.main()
        self.load_dataset()


    def train_test_split(self):
        self.df = self.df.dropna()
        X, self.X_test, y, self.y_test = train_test_split(self.df, self.df["sentiment"],
                                                                                test_size=self.parameters["test_size"], random_state=42)

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y,
                                                                                  random_state=42,
                                                                                  test_size=self.parameters["valid_size"])

    def get_features(self):
        features = Features()
        features.generate_vectorizer(max_features=self.parameters["max_features"], ngram_range=self.parameters["ngrams"])
        self.X_train = features.fit_transform(self.X_train)
        self.X_valid = features.transform(self.X_valid)
        self.X_test = features.transform(self.X_test)


    def classify(self):
        if parameters["classifier"] == "NB":
            nb = NaiveBayes.NaiveBayes()
            nb.fit(self.X_train, self.y_train)
            Evaluator.evaluate_classifier(self.y_valid, nb.predict(self.X_valid))



    def run(self):
        if self.parameters["preprocess"]:
            self.preprocess()

        if self.parameters["train_test_split"]:
            self.train_test_split()

        self.get_features()

        if self.parameters["pca"]:
            pass  # da implementare

        self.classify()







if __name__ == '__main__':
    parameters = {
        "dataset" : "Sentiment140.csv",
        "preprocess": False,
        "train_test_split": True,
        "ngrams": (1, 1),
        "max_features": None,
        "pca": False,
        "classifier": "NB",
        "valid_size": 0.3,
        "test_size": 0.2
    }

    main = Main(parameters)
    main.run()
