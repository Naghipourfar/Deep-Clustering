import pandas as pd

"""
    Created by Mohsen Naghipourfar on 9/6/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""
mer_num = [3, 5]
types = ["", "_WGS", "_WXS"]
for mer in mer_num:
    for type in types:
        x = pd.read_csv("../Data/" + str(mer) + "mermotif" + type + ".tsv", delimiter="\t", index_col="icgc_sample_id")
        # x.dropna(how="all", axis=1, inplace=True)
        # x.dropna(how="any", axis=1, inplace=True)
        print(str(mer) + "mer" + type + "\t:\t" + str(x.shape))
