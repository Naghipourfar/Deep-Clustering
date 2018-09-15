import pandas as pd

from sklearn.cluster import DBSCAN

"""
    Created by Mohsen Naghipourfar on 9/6/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""

x = pd.read_csv("../Data/3mermotif.tsv", delimiter="\t", index_col="icgc_sample_id")
x.dropna(how="all", axis=1, inplace=True)
x.dropna(how="any", axis=1, inplace=True)
print(x.shape)

x.to_csv("../Data/3mermotif_na.csv")
# dbscan = DBSCAN(eps=0.3)
# y_pred = dbscan.fit_predict(x.values)
# print(y_pred.shape)
# print(pd.Series(y_pred).value_counts())
# print(pd.DataFrame(y_pred).describe())
#


