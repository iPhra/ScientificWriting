from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCA_features:

    def __init__(self):
        self.pca = None
        self.scaler = StandardScaler(with_std=False)

    def generate_vectorizer(self, n_components):
        self.pca = PCA(n_components)

    def fit_transform(self, train):
        train = self.scaler.fit_transform(train)
        return self.pca.fit_transform(train)

    def transform(self, test):
        test = self.scaler.transform(test)
        return self.pca.transform(test)
