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

import chainer
from chainer import cuda, Variable
from chainer.dataset import convert
from chainer import serializers
from chainer.datasets import get_cifar10

from mllogger import MLLogger
from models import small
from models import medium
logger = MLLogger()

def get_color_map_nipy(gradation_num):
    colors = []
    for idx in [x*255/gradation_num for x in xrange(gradation_num)]:
        colors.append(plt.cm.nipy_spectral(idx)[0:3])
    return (np.array(colors)[::-1,(2,1,0)]*255).astype(np.int)

def plot_nn(data, label, nn_result, nn_result2, k):
    colors = get_color_map_nipy(10)
    size = 32
    canvas = np.zeros((size * 100, size * k * 2, 3), dtype=np.uint8)

    # (32x32を縦に配置)
    dists, nearest_idxs = nn_result
    for i, idxs in enumerate(nearest_idxs[:100]):
        idx = idxs[0]
        canvas[i * size:(i + 1) * size, 0:size] = \
            np.transpose(data[idx], (1, 2, 0))[:, :, ::-1] * 255
        cv2.rectangle(canvas, (0, i * size), (size - 1, (i + 1) * size - 1),
                      colors[label[idx]], 1)
    for i, idxs in enumerate(nearest_idxs[:100]):
        for j, idx in enumerate(idxs[1:]):
            x1 = int((1.5 + j) * size)
            x2 = int((2.5 + j) * size)
            canvas[i * size:(i + 1) * size, x1:x2] = \
                np.transpose(data[idx], (1, 2, 0))[:, :, ::-1] * 255
            cv2.rectangle(canvas, (x1, i * size), (x2 - 1, (i + 1) * size - 1),
                          colors[label[idx]], 1)
    dists, nearest_idxs = nn_result2
    for i, idxs in enumerate(nearest_idxs[:100]):
        for j, idx in enumerate(idxs[1:]):
            x1 = int((k + 1 + j) * size)
            x2 = int((k + 2 + j) * size)
            canvas[i * size:(i + 1) * size, x1:x2] = \
                np.transpose(data[idx], (1, 2, 0))[:, :, ::-1] * 255
            cv2.rectangle(canvas, (x1, i * size), (x2 - 1, (i + 1) * size - 1),
                          colors[label[idx]], 1)
    return canvas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # NN
    parser.add_argument('--nb_samples', type=int, default=1000)
    parser.add_argument('--nn_jobs', type=int, default=12)
    parser.add_argument('--nb_neighbors', type=int, default=10)

    # Model
    parser.add_argument('--model', default='c3f2')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--resume', '-r', default='')
    parser.add_argument('--N',  type=int, default=9)
    parser.add_argument('--k',  type=int, default=10)

    # Others
    parser.add_argument('--seed', type=int, default=1701)
    args = parser.parse_args()
    logger.info(__file__)
    logger.info(vars(args))
    start = time.time()
    np.random.seed(args.seed)

    if args.model == "c3f2":
        model = small.c3f2()
    elif args.model == "fconv":
        model = small.fconv()
    elif args.model == "mlp":
        model = small.MLP()
    elif args.model == "wideresnet":
        model = medium.WideResNet(args.N, args.k)
    serializers.load_npz(args.resume, model)
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    train, _ = get_cifar10()

    if args.nb_samples != -1:
        choice = np.random.choice(len(train), args.nb_samples, replace=False)
        data, label = train[choice, ...]
    else:
        data, label = train[range(len(train)), ...]

    predictions = []
    chainer.config.train = False
    chainer.config.enable_backprop = False
    for im in data:
        x = Variable(cuda.to_gpu(im[np.newaxis, ...], args.gpu))
        predictions.append(model.get_feature(x).data)
    predictions = np.array([cuda.to_cpu(x) for x in predictions])

    nn = NearestNeighbors(n_neighbors=args.nb_neighbors, algorithm="ball_tree", n_jobs=args.nn_jobs)
    neighbors = nn.fit(predictions)
    nn_result = neighbors.kneighbors(predictions)

    nn = NearestNeighbors(n_neighbors=args.nb_neighbors, algorithm="ball_tree", n_jobs=args.nn_jobs)
    neighbors = nn.fit(data.reshape((len(data), -1)))
    nn_result2 = neighbors.kneighbors(data.reshape((len(data), -1)))

    canvas = plot_nn(data, label, nn_result, nn_result2, args.nb_neighbors)
    cv2.imwrite("{}_nn_deep.jpg".format(args.model), canvas)

    print(time.time()-start)
