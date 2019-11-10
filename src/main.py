import pandas as pd
from src import preprocess
from src.features.pca import PCA_features
from src.features.n_grams import N_grams
from src.classifiers import NaiveBayes, Classifier, SVM, RandomForest, Evaluator
from sklearn.model_selection import train_test_split


class Main:

    def __init__(self, commands, parameters):
        self.parameters = parameters
        self.commands = commands
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
        features = N_grams()
        features.generate_vectorizer(max_features=self.parameters["max_features"], ngram_range=self.parameters["ngrams"])
        self.X_train = features.fit_transform(self.X_train)
        self.X_valid = features.transform(self.X_valid)
        self.X_test = features.transform(self.X_test)
        print("After n-gram features, size of train-validation-test:")
        print(self.X_train.shape)
        print(self.X_valid.shape)
        print(self.X_test.shape)
        print("\n")

    def pca(self):
        pca = PCA_features()
        pca.generate_vectorizer(n_components=1000)
        self.X_train = pca.fit_transform(self.X_train.todense())
        self.X_valid = pca.transform(self.X_valid.todense())
        self.X_test = pca.transform(self.X_test.todense())
        print("After PCA, size of train-validation-test:")
        print(self.X_train.shape)
        print(self.X_valid.shape)
        print(self.X_test.shape)
        print("\n")

    def classify(self):
        classifiers = []
        if "NB" in parameters["classifier"]:
            classifiers.append(NaiveBayes.NaiveBayes())
        if "RF" in parameters["classifier"]:
            classifiers.append(RandomForest.RandomForest())
        if "SVM" in parameters["classifier"]:
            classifiers.append(SVM.SVM())

        for classifier in classifiers:
            print("Training "+str(classifier))
            classifier.fit(self.X_train, self.y_train)
            Evaluator.evaluate_classifier(self.y_valid, classifier.predict(self.X_valid))
            print("\n")


    def run(self):
        if self.commands["preprocess"]:
            self.preprocess()

        if self.commands["train_test_split"]:
            self.train_test_split()

        self.get_features()

        if self.commands["pca"]:
            self.pca()

        if self.commands["classify"]:
            self.classify()







if __name__ == '__main__':
    parameters = {
        "dataset": "Sentiment140.csv",
        "ngrams": (1, 1),
        "max_features": None,
        "classifier": ["RF"],
        "valid_size": 0.3,
        "test_size": 0.2
    }

    commands = {
        "preprocess": False,
        "train_test_split": True,
        "pca": True,
        "classify": True
    }

    main = Main(commands, parameters)
    main.run()
