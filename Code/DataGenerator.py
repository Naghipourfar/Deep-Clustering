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
RADIUS = 10


def draw_ring(n_points=250):
    xs, ys = [], []
    r = random.random() * RADIUS
    for i in range(n_points):
        p = 2 * math.pi * random.random()
        x = r * math.cos(p)
        y = r * math.sin(p)
        x += 1
        y += 1
        xs.append(x)
        ys.append(y)
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


def rotate_point(xs, ys, degree, origin=(0, 0)):
    degree = math.radians(degree)
    for idx, x in enumerate(xs):
        y = ys[idx]
        ox, oy = origin
        x -= ox
        y -= oy
        cos = math.cos(degree)
        sin = math.sin(degree)
        rotated_x = cos * x - sin * y
        rotated_y = sin * x + cos * y
        xs[idx], ys[idx] = rotated_x + ox, rotated_y + oy
    return xs, ys


def plot_shapes(xs, ys):
    plt.close("all")
    plt.figure(figsize=(512 / MY_DPI, 512 / MY_DPI), dpi=MY_DPI)
    # plt.figure(figsize=(15, 10))
    plt.plot(xs, ys, 'o')
    # plt.axis('off')
    plt.grid()
    # plt.xlim((-2, 2))
    # plt.ylim((-2, 2))
    plt.savefig("../Results/image.png")


def main():
    number_of_objects = 10
    shapes = ['square ring', 'ring']
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
            degree = random.random() * 180
            origin = random.random() * 2 - 1, random.random() * 2 - 1
            x, y = rotate_point(x, y, degree, origin)
            xs = xs + x
            ys = ys + y
    plot_shapes(xs, ys)
    return xs, ys


if __name__ == '__main__':
    # x, y = [1], [1]
    # degree = 45
    # print(rotate_point(x, y, degree))
    main()
