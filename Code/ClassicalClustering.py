import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyclustering.cluster import kmedians
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.mixture import GMM
from sklearn.preprocessing import LabelEncoder, normalize
import seaborn as sns
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
    return km


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


def make_mapping_matrix(table):
    df = pd.DataFrame()
    # print(table["prediction"].describe())
    cancer_types = table["cancer_type"].unique()
    for cancer_type in cancer_types:
        x = table[table["cancer_type"] == cancer_type]
        for i in range(table["prediction"].max() + 1):
            y = x[x["prediction"] == i]
            df.loc[cancer_type, i] = len(y)
    return df


def plot_heatmaps(mapped_table, alg_name=None):
    cancer_types = mapped_table.transpose().columns
    cluster_labels = mapped_table.columns
    mapped_table = np.array(mapped_table)
    plt.close("all")

    fig, ax = plt.subplots()
    im = ax.imshow(mapped_table.transpose())

    ax.set_xticks(np.arange(len(cancer_types)))
    ax.set_yticks(np.arange(len(cluster_labels)))
    ax.set_xticklabels(cancer_types)
    ax.set_yticklabels(cluster_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax.set_xticks(np.arange(mapped_table.shape[0] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(mapped_table.shape[1] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    plt.title("Results for %s" % alg_name)
    plt.xlabel("Cancer Types")
    plt.ylabel("Cluster Labels")
    plt.tight_layout()
    plt.savefig("../Results/Presentation/Heatmaps/%s.pdf" % alg_name, bbox_inches="tight")
    print("Heatmap for %s is finished" % alg_name)


def plot_stacks(mapped_table, alg_name=None):
    x = mapped_table
    N = len(x.index)
    width = .95
    xs = [i for i in range(N)]
    # xs = list(x.transpose().columns)
    plt.figure(figsize=(15, 10))
    for cluster in x.columns:
        plt.bar(xs, x.loc[:, cluster], width=width, bottom=None, label=cluster)
    plt.legend()
    plt.xticks(xs, x.transpose().columns, rotation=90)
    plt.title("Results for %s" % alg_name)
    plt.xlabel("Cancer Types")
    plt.ylabel("Sample Frequency")
    plt.tight_layout()
    plt.savefig("../Results/Presentation/Stacked Barplots/%s.pdf" % alg_name, bbox_inches="tight")
    print("Stacked Bar plot for %s is finished" % alg_name)


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

    for alg_name in ["Kmeans", "GMM", "DBSCAN", "HDBSCAN"]:
        for k in range(2, 24):
            if alg_name == "Kmeans":
                predictions = k_means(data, k)
                new_alg_name = alg_name + " (k = %d)" % k
            elif alg_name == "GMM":
                predictions = gaussian_mixture_models(data, k)
                new_alg_name = alg_name + " (k = %d)" % k
            elif alg_name == "KMediods":
                predictions = k_medians(data, k)
                new_alg_name = alg_name + " (k = %d)" % k
            else:
                if k > 2:
                    continue
                if alg_name == "DBSCAN":
                    min_pts = 5
                    eps = .3
                    predictions = dbscan(data)
                    new_alg_name = alg_name + " (minPts = %d, eps = %.1f)" % (min_pts, eps)
                else:
                    min_cluster_size = 5
                    predictions = hdbscan_algorithm(data)
                    new_alg_name = alg_name
            table = pd.DataFrame({"cancer_type": cancer_types, "prediction": predictions})
            if alg_name.endswith("SCAN"):
                table.loc[:, 'prediction'] += 1
            mapped_table = make_mapping_matrix(table)
            plot_stacks(mapped_table, new_alg_name)
            plot_heatmaps(mapped_table, new_alg_name)
