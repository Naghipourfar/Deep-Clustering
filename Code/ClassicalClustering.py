import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GMM
from sklearn.preprocessing import LabelEncoder, normalize

"""
    Created by Mohsen Naghipourfar on 9/15/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""


def dbscan(data, minimum_points=5, eps=0.7):
    dbscan = DBSCAN(eps=eps, min_samples=minimum_points)
    y_pred = dbscan.fit_predict(data)
    return y_pred


def pca_analysis(data, n_components=2):
    pca = PCA(n_components=n_components)
    x = pca.fit_transform(data)
    return x[:, 0], x[:, 1]


def ica_analysis(data, n_components=2):
    ica = FastICA(n_components=n_components)
    x = ica.fit_transform(data)
    return x[:, 0], x[:, 1]


def hdbscan_algorithm(min_cluster_size=10):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    cluster_labels = clusterer.fit_predict(data)
    return cluster_labels


def model_based():
    pass


def k_means(data, k=10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    y_kmeans = kmeans.predict(data)
    return y_kmeans


def k_medoids(k=10):
    pass


def k_medians(k=10):
    pass


def gaussian_mixture_models(data, k=10):
    gmm = GMM(n_components=k).fit(data)
    labels = gmm.predict(data)
    return labels


def plot_points(data, labels, predictions=None):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    x, y = pca_analysis(data, n_components=2)
    plt.close("all")
    plt.scatter(x, y, c=labels, s=50, cmap='viridis')
    plt.show()
    # plt.savefig("/Users/Future/Desktop/test.pdf")


def normalize_data(data):
    data = normalize(data, axis=1, norm='l2')
    return data


def make_table(table):
    unique_cancer_types = table['cancer_type'].unique()
    for cancer_type in unique_cancer_types:
        x = table[table['cancer_type'] == cancer_type]
        n_cancer_type = x.shape[0]
        n_frequent_cluster = pd.value_counts(x["prediction"].values, sort=True, ascending=False).iloc[0]
        cluster_num = pd.value_counts(x["prediction"].values, sort=True, ascending=False).index[0]

        p_right_prediction = float(n_frequent_cluster / n_cancer_type) * 100
        print("%s & %d & %d & %d & %.2f " % (
        cancer_type, cluster_num, n_cancer_type, n_frequent_cluster, p_right_prediction) + "\\% \\\\")


if __name__ == '__main__':
    data = pd.read_csv("../Data/3mermotif_na.csv", index_col="icgc_sample_id")
    cancer_types = data['cancer_type']
    data = data.drop(["cancer_type"], axis=1)
    data = normalize_data(data.values)
    # plot_points(data, cancer_types)
    predictions = k_means(data, k=23)
    table = pd.DataFrame({"cancer_type": cancer_types, "prediction": predictions})
    make_table(table)
    # print(y_pred.describe())
