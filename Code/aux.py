import numpy as np

"""
    Created by Mohsen Naghipourfar on 10/4/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""

import pandas as pd

data = pd.read_csv("../Data/5mermotif.csv", index_col="icgc_sample_id")
cancer_types = data["cancer_type"]
data.drop("cancer_type", axis=1, inplace=True)
data = data.values

np.savetxt(fname="../Data/5mermotif.txt", X=data, delimiter="\t")
