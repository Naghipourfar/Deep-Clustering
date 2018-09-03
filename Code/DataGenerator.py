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
    # plt.plot(xs, ys, 'o')
    # plt.show()
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
    # plt.close("all")
    # plt.figure(figsize=(512 / MY_DPI, 512 / MY_DPI), dpi=MY_DPI)
    # plt.plot(xs, ys, 'o')
    # plt.axis("off")
    # plt.savefig("../Results/image.png")
    return xs, ys


def draw_square_ring(n_points=250):
    xs, ys = [], []
    while len(xs) <= n_points:
        x = random.random()
        y = random.random()
        std = 0.05
        if abs(x - 1) <= std:
            xs.append(x)
            ys.append(y)
        elif abs(x) <= std:
            xs.append(x)
            ys.append(y)
        elif abs(y) <= std:
            ys.append(y)
            xs.append(x)
        elif abs(y - 1) <= std:
            ys.append(y)
            xs.append(x)
    return xs, ys


def rotate_point(xs, ys, degree):
    for idx, x in enumerate(xs):
        y = xs[idx]
        degree = math.radians(degree)
        ox, oy = 1, 1
        rotated_x = ox + math.cos(degree) * (x - ox) - math.sin(degree) * (y - oy)
        rotated_y = oy + math.sin(degree) * (x - ox) + math.cos(degree) * (y - oy)
        xs[idx], ys[idx] = rotated_x, rotated_y
    return xs, ys


def plot_shapes(xs, ys):
    plt.close("all")
    # plt.figure(figsize=(512 / MY_DPI, 512 / MY_DPI), dpi=MY_DPI)
    plt.figure(figsize=(15, 10))
    plt.plot(xs, ys, 'o')
    # plt.axis('off')
    plt.savefig("../Results/image.png")


if __name__ == '__main__':
    number_of_objects = 1
    shapes = ['circle']
    xs, ys = [], []
    for shape in shapes:
        for i in range(random.randint(1, number_of_objects)):
            x, y = [], []
            if shape == "circle":
                x, y = draw_circle(n_points=300)
            elif shape == "square ring":
                x, y = draw_square_ring(n_points=300)
            elif shape == "ring":
                x, y = draw_ring(n_points=300)
            x, y = rotate_via_numpy(x, y, degree=random.random() * 180)
            xs = xs + x
            ys = ys + y
    plot_shapes(xs, ys)
