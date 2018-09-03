import math
import random

import matplotlib.pyplot as plt
import numpy as np

"""
    Created by Mohsen Naghipourfar on 9/2/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""



def draw_circle(n_points=500):
    xs, ys = [], []
    for i in range(0, n_points):
        p = random.random() * 2 * math.pi
        r = 1 * math.sqrt(0.01)
        x = math.cos(p) * r
        y = math.sin(p) * r
        xs.append(x)
        ys.append(y)
    plt.plot(xs, ys, 'o')
    plt.show()


draw_circle()
