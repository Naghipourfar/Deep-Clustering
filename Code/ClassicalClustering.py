import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyclustering.cluster import kmedians
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


def dbscan(data, minimum_points=5, eps=0.3):
    dbscan = DBSCAN(eps=eps, min_samples=minimum_points)
    y_pred = dbscan.fit_predict(data)
    return y_pred


def pca_analysis(data, n_components=2):
    pca = PCA(n_components=n_components)
    x = pca.fit_transform(data)
    if n_components == 2:
        return x[:, 0], x[:, 1]
    if n_components == 3:
        return x[:, 0], x[:, 1], x[:, 2]


def ica_analysis(data, n_components=2):
    ica = FastICA(n_components=n_components)
    x = ica.fit_transform(data)
    if n_components == 2:
        return x[:, 0], x[:, 1]
    if n_components == 3:
        return x[:, 0], x[:, 1], x[:, 2]


def hdbscan_algorithm(data, min_cluster_size=18):
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


def k_medians(data, k=10):
    km = kmedians.kmedians(data, np.random.normal(0.0, 0.1, size=[k, data.shape[1]]))
    km.process()
    print(km.get_clusters())


def gaussian_mixture_models(data, k=10):
    gmm = GMM(n_components=k).fit(data)
    labels = gmm.predict(data)
    return labels


def plot_points(data, labels, predictions=None):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    x, y, z = pca_analysis(data, n_components=3)
    # plt.close("all")
    # plt.xlabel("1st PCA")
    # plt.ylabel("2nd PCA")
    # plt.scatter(x, y, c=labels, s=50, cmap='viridis')
    # plt.savefig("/Users/Future/Desktop/PCA.pdf")
    # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z)
    ax.set_xlim(0, x.max())
    ax.set_ylim(0, y.max())
    ax.set_zlim(0, z.max())

    plt.savefig("/Users/Future/Desktop/test.pdf")
    plt.show()


def plot_stacks(table, alg_name=None):
    df = pd.DataFrame()
    print(table["prediction"].describe())
    cancer_types = table["cancer_type"].unique()
    for cancer_type in cancer_types:
        x = table[table["cancer_type"] == cancer_type]
        for i in range(table["prediction"].max() + 1):
            y = x[x["prediction"] == i]
            df.loc[cancer_type, i] = len(y)
    x = df
    N = len(x.index)
    width = .95
    xs = [i for i in range(N)]
    # xs = list(x.transpose().columns)
    plt.figure(figsize=(15, 10))
    for cluster in x.columns:
        plt.bar(xs, x.loc[:, cluster], width=width, bottom=None, label=cluster)
    # plt.bar(xs, x['C'].iloc[:N], width=width, bottom=x['A'].iloc[:N], label="C")
    # plt.bar(xs, x['G'].iloc[:N], width=width, bottom=x['A'].iloc[:N] + x['C'].iloc[:N], label="G")
    # plt.bar(xs, x['T'].iloc[:N], width=width, bottom=x['A'].iloc[:N] + x['C'].iloc[:N] + x['G'].iloc[:N], label="T")
    plt.legend()
    plt.xticks(xs, x.transpose().columns, rotation=90)
    plt.title("Results for %s" % alg_name)
    plt.xlabel("Cancer Types")
    plt.ylabel("Sample Frequency")
    plt.savefig("../Results/%s_transposed.pdf" % alg_name)


def normalize_data(data):
    data = normalize(data, axis=0, norm='max')
    return data


def make_table(table, filename=None, write=True):
    print("\\begin{table}[h!]")
    print("\\resizebox{\columnwidth}{!}{")
    print("\\centering")
    print("\\begin{tabular}{||c c c c c||}")
    print("\\hline")
    print("Cancer & Cluster & \# Samples & \# Cluster & Percentage \\\\ [0.5ex]")
    print("\\hline\\hline")
    unique_cancer_types = table['cancer_type'].unique()

    df = pd.DataFrame()
    for cancer_type in unique_cancer_types:
        x = table[table['cancer_type'] == cancer_type]
        n_cancer_type = x.shape[0]
        value_counts = pd.value_counts(x["prediction"].values, sort=True, ascending=False)
        n_frequent_cluster, cluster_num = (value_counts.iloc[0], value_counts.index[0]) if value_counts.index[
                                                                                               0] != -1 else (
            value_counts.iloc[1], value_counts.index[1])
        p_right_prediction = float(n_frequent_cluster / n_cancer_type) * 100
        df[cancer_type] = [p_right_prediction]
        if cancer_type == "Head&Neck":
            cancer_type = "Head\\&Neck"
        elif cancer_type == "Nervous_system":
            cancer_type = "Nervous\\_System"
        print("%s & %d & %d & %d & %.2f " % (
            cancer_type, cluster_num, n_cancer_type, n_frequent_cluster, p_right_prediction) + "\\% \\\\")
        # print(pd.value_counts(x["prediction"].values, sort=True, ascending=False))
        # print("*" * 50)

    print("\\hline")
    print("\\end{tabular}")
    print("}")
    print("\caption{%s}" % filename)
    print("\\end{table}")
    print(end="\n\n\n\n")
    filename = "../Results/classic/" + filename + ".csv"
    if write:
        df.to_csv(filename)


if __name__ == '__main__':
    data = pd.read_csv("../Data/3mermotif_na.csv", index_col="icgc_sample_id")
    cancer_types = data['cancer_type']
    data = data.drop(["cancer_type"], axis=1)
    data = normalize_data(data.values)

    # plot_points(data, cancer_types)
    for k in range(18, 19):
        predictions = hdbscan_algorithm(data)
        table = pd.DataFrame({"cancer_type": cancer_types, "prediction": predictions})
        table.loc[:, 'prediction'] += 1
        plot_stacks(table, "HDBSCAN")
    # print('Accuracy: {}'.format(accuracy_score(cancer_types, predictions, normalize=False)))
    # print(y_pred.describe())
