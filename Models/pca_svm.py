from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import numpy as np


class classification:

    def __init__(self, features, lables)

        self.features = features
        self.labels = labels

    def dim_reduction(self, dimensions):

        pca = PCA(n_components=dimensions)
        feature_pca = pca.fit_transform(self.features)

        clf = LinearSVC()
        clf.fit(feature_pca, labels)

        preds = clf.predict(feature_pca)

        return preds
