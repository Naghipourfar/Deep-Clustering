import math
import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

"""
    Created by Mohsen Naghipourfar on 9/2/18.
    Email : mn7697np@gmail.com or naghipourfar@ce.sharif.edu
    Website: http://ce.sharif.edu/~naghipourfar
    Github: https://github.com/naghipourfar
    Skype: mn7697np
"""
MY_DPI = 192
RADIUS = 10
WIDTH = 10


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
    width = random.random() * WIDTH
    std = 0.025
    np.random.seed(int(n_points * width))
    x = list(np.random.normal(loc=0.0, scale=std, size=[n_points // 4]))
    x += list(np.random.uniform(0.0, width, size=[n_points // 4]))
    x += list(np.random.normal(loc=width, scale=std, size=[n_points // 4]))
    x += list(np.random.uniform(0.0, width, size=[n_points // 4]))

    np.random.seed(int(n_points * width + 1))
    y = list(np.random.uniform(0.0, width, size=[n_points // 4]))
    y += list(np.random.normal(loc=width, scale=std, size=[n_points // 4]))
    y += list(np.random.uniform(0.0, width, size=[n_points // 4]))
    y += list(np.random.normal(loc=0.0, scale=std, size=[n_points // 4]))

    return x, y


def rotate_point(xs, ys, degree, origin=(0, 0)):
    degree = math.radians(degree)
    bias = random.random() * 5
    for idx, x in enumerate(xs):
        y = ys[idx]
        ox, oy = origin
        x -= ox
        y -= oy
        cos = math.cos(degree)
        sin = math.sin(degree)
        rotated_x = cos * x - sin * y
        rotated_y = sin * x + cos * y
        xs[idx], ys[idx] = rotated_x + ox + bias, rotated_y + oy + bias
    return xs, ys


def plot_shapes(xs, ys, idx=0):
    result_path = "../Results/train/" + "image-" + str(idx) + ".png"
    plt.axis('off')
    plt.savefig(result_path, dpi=MY_DPI)
    data = plt.imread(result_path)[:, :, 0]
    plt.imsave(result_path, data, cmap=cm.gray)


def main():
    for idx in range(100):
        plt.close("all")
        plt.figure(figsize=(128 / MY_DPI, 128 / MY_DPI), dpi=MY_DPI)
        number_of_objects = 5
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
                x, y = rotate_point(x, y, degree, origin=(0, 0))
                xs = xs + x
                ys = ys + y
        plt.plot(xs, ys, 'o')
        plot_shapes(xs, ys, idx)


if __name__ == '__main__':
    # x, y = [1], [1]
    # degree = 45
    # print(rotate_point(x, y, degree))
    main()
