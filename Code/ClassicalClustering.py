import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

"""
    Created by Mohsen Naghipourfar on 9/15/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def dbscan(data, minimum_points=5, eps=0.7):
    dbscan = DBSCAN(eps=eps, min_samples=minimum_points)
    y_pred = dbscan.fit_predict(data.values)
    return y_pred


def pca_analysis(data, n_components=2):
    pca = PCA(n_components=n_components)
    x = pca.fit_transform(data.values)
    return x[:, 0], x[:, 1]


def hdbscan():
    pass


def model_based():
    pass


def k_means(k=10):
    pass


def k_medoids(k=10):
    pass


def gaussian_mixture_models(k=10):
    pass


def plot_points(data, labels, predictions=None):
    pass



if __name__ == '__main__':
    data = pd.read_csv("../Data/3mermotif_na.csv", index_col="icgc_sample_id")
    cancer_types = data['cancer_type']
    data = data.drop(["cancer_type"], axis=1)
    y_pred = dbscan(data)
    y_pred = pd.DataFrame(y_pred)
    print(y_pred.describe())
