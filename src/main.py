import pandas as pd
from src import preprocess
from src.features.pca import PCA_features
from src.features.n_grams import N_grams
from src.features.tfidf import Tfidf
from src.classifiers import NaiveBayes, SVM, RandomForest, Evaluator
from sklearn.model_selection import train_test_split
from random import randint


class Main:

    def __init__(self, com, par):
        self.parameters = par
        self.commands = com
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
        print("Preprocessing...")
        preprocess.main()
        self.load_dataset()
        print("Finished preprocessing\n")


    def train_test_split(self):
        self.df = self.df.dropna()
        X, self.X_test, y, self.y_test = train_test_split(self.df, self.df["sentiment"],
                                                            test_size=self.parameters["test_size"], random_state=self.parameters["seed"])

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y,
                                                                                  random_state=self.parameters["seed"],
                                                                                  test_size=self.parameters["valid_size"])

    def get_features(self, tfidf=False):
        print("Extracting features...")
        if tfidf:
            features = Tfidf()
        else:
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
        print("Performing PCA...")
        pca = PCA_features()
        pca.generate_vectorizer(self.parameters["pca_components"])
        self.X_train = pca.fit_transform(self.X_train.todense())
        self.X_valid = pca.transform(self.X_valid.todense())
        self.X_test = pca.transform(self.X_test.todense())
        print("After PCA, size of train-validation-test:")
        print(self.X_train.shape)
        print(self.X_valid.shape)
        print(self.X_test.shape)
        print("PCA applied\n")


    def classify(self):
        result = []
        classifiers = []
        if "NB" in parameters["classifier"]:
            classifiers.append(NaiveBayes.NaiveBayes(self.parameters["nb_alpha"]))
        if "RF" in parameters["classifier"]:
            classifiers.append(RandomForest.RandomForest(self.parameters["rf_components"]))
        if "SVM" in parameters["classifier"]:
            classifiers.append(SVM.SVM())

        for classifier in classifiers:
            print("Training "+str(classifier))
            classifier.fit(self.X_train, self.y_train)
            result.append(Evaluator.evaluate_classifier(self.y_valid, classifier.predict(self.X_valid)))
            print("\n")
        return max(result)


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


    def compare_ngrams_rf(self):
        res = []
        for components in (10, 100, 500):
            for i in range(1, 2):
                for j in range(i, 6):
                    print("Training with ngrams: ", (i, j))
                    self.parameters["ngrams"] = (i, j)
                    self.train_test_split()
                    self.get_features(tfidf=True)
                    rf = RandomForest.RandomForest(n=components)
                    rf.fit(self.X_train, self.y_train)
                    accuracy = Evaluator.evaluate_classifier(self.y_valid, rf.predict(self.X_valid))
                    res.append(((i, j, components), accuracy))
        print(sorted(res, key=lambda x: x[1], reverse=True))


    def compare_ngrams_nb(self):
        res = []
        for a in (1, 10, 100):
            for i in range(1, 2):
                for j in range(i, 7):
                    print("Training with ngrams: ", (i, j))
                    self.parameters["ngrams"] = (i, j)
                    self.train_test_split()
                    self.get_features(tfidf=True)
                    nb = NaiveBayes.NaiveBayes(alpha=a)
                    nb.fit(self.X_train, self.y_train)
                    accuracy = Evaluator.evaluate_classifier(self.y_valid, nb.predict(self.X_valid))
                    res.append(((i, j, a), accuracy))
        print(sorted(res, key=lambda x: x[1], reverse=True))


    def compare_ngrams_svm(self):
        res = []
        for c in (0.1, 1, 10):
            for i in range(1, 2):
                for j in range(i, 7):
                    print("Training with ngrams: ", (i, j))
                    self.parameters["ngrams"] = (i, j)
                    self.train_test_split()
                    self.get_features(tfidf=True)
                    svm = SVM.SVM(c=c)
                    svm.fit(self.X_train, self.y_train)
                    accuracy = Evaluator.evaluate_classifier(self.y_valid, svm.predict(self.X_valid))
                    res.append(((i, j, c), accuracy))
        print(sorted(res, key=lambda x: x[1], reverse=True))





if __name__ == '__main__':
    parameters = {
        "dataset": "Sentiment140.csv",
        "ngrams": (1, 5),
        "max_features": None,
        "classifier": ["RF"],
        "valid_size": 0.3,
        "test_size": 0.2,
        "rf_components": 100,
        "pca_components": 5000,
        "nb_alpha": 1,
        "seed": randint(1, 100)
    }

    commands = {
        "preprocess": False,
        "train_test_split": True,
        "pca": False,
        "classify": True
    }

    main = Main(commands, parameters)
    #main.run()
    main.compare_ngrams_svm()
