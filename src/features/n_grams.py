from sklearn.feature_extraction.text import CountVectorizer


class N_grams:

    def __init__(self):
        self.vectorizer = None

    def generate_vectorizer(self, max_features=None, min_df=2, max_df=0.90, ngram_range=(1, 1)):
        self.vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, max_features=max_features,
                                          ngram_range=ngram_range)

    def fit_transform(self, df):
        return self.vectorizer.fit_transform(df["tweet"])

    def transform(self, df):
        return self.vectorizer.transform(df["tweet"])
