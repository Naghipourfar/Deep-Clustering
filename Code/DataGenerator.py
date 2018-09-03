import math
import random

import matplotlib.pyplot as plt

"""
    Created by Mohsen Naghipourfar on 9/2/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""
MY_DPI = 192


def draw_ring(n_points=250):
    xs, ys = [], []
    for i in range(n_points):
        p = random.random() * 2 * math.pi
        r = 1 * math.sqrt(0.01)
        x = math.cos(p) * r
        y = math.sin(p) * r
        xs.append(x)
        ys.append(y)
    plt.plot(xs, ys, 'o')
    plt.show()
    return xs, ys


def draw_circle(n_points=250):
    xs, ys = [], []
    for i in range(n_points):
        p = random.random() * 2 * math.pi
        r = 1 * math.sqrt(random.random())
        x = math.cos(p) * r
        y = math.sin(p) * r
        xs.append(x)
        ys.append(y)
    plt.close("all")
    plt.figure(figsize=(512 / MY_DPI, 512 / MY_DPI), dpi=MY_DPI)
    plt.plot(xs, ys, 'o')
    plt.axis("off")
    plt.savefig("../Results/image.png")
    return xs, ys


def draw_square_ring(n_points=250):
    xs, ys = [], []
    for i in range(n_points):
        x = random.random()
        y = random.random()
        xs.append(x)
        ys.append(y)
    plt.close("all")
    plt.figure(figsize=(512 / MY_DPI, 512 / MY_DPI), dpi=MY_DPI)
    plt.plot(xs, ys, 'o')
    plt.axis('off')
    plt.savefig("../Results/image.png")
    return xs, ys

draw_circle(n_points=500)
