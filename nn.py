#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import os
import sys
import argparse
import time

import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from chainer.datasets import get_cifar10

from mllogger import MLLogger
logger = MLLogger()

def get_color_map_nipy(gradation_num):
    colors = []
    for idx in [x*255/gradation_num for x in xrange(gradation_num)]:
        colors.append(plt.cm.nipy_spectral(idx)[0:3])
    return (np.array(colors)[::-1,(2,1,0)]*255).astype(np.int)

def plot_nn(data, label, nn_result, k):
    dists, nearest_idxs = nn_result
    colors = get_color_map_nipy(10)

    # (32x32を縦に配置)
    size = 32
    canvas = np.zeros((size * 100, size * k, 3), dtype=np.uint8)
    for i, idxs in enumerate(nearest_idxs[:100]):
        for j, idx in enumerate(idxs):
            canvas[i * size:(i + 1) * size, j * size:(j + 1) * size] = \
                np.transpose(data[idx], (1, 2, 0))[:, :, ::-1] * 255
            cv2.rectangle(canvas,
                          (j * size, i * size),
                          ((j + 1) * size - 1, (i + 1) * size - 1),
                          colors[label[idx]], 1)
    return canvas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_samples', type=int, default=1000)
    parser.add_argument('--nn_jobs', type=int, default=12)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1701)
    args = parser.parse_args()
    logger.info(__file__)
    logger.info(vars(args))
    start = time.time()
    np.random.seed(args.seed)

    train, _ = get_cifar10()

    if args.nb_samples != -1:
        choice = np.random.choice(len(train), args.nb_samples, replace=False)
        data, label = train[choice, ...]
    else:
        data, label = train[range(len(train)), ...]

    nn = NearestNeighbors(n_neighbors=args.k, algorithm="ball_tree", n_jobs=args.nn_jobs)
    neighbors = nn.fit(data.reshape((len(data), -1)))
    nn_result = neighbors.kneighbors(data.reshape((len(data), -1)))

    canvas = plot_nn(data, label, nn_result, args.k)
    cv2.imwrite("nn.jpg", canvas)

    print(time.time()-start)
