#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

from __future__ import print_function
from __future__ import division
from six.moves import range

import os
import time
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import chainer
from chainer.dataset import convert
from chainer import serializers
from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

from models import small
from models import medium

def main():
    """
    学習済みモデルをロードしてフィルタを見る
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('resume')
    parser.add_argument('--model', default='c3f2')
    parser.add_argument('--N',  type=int, default=9)
    parser.add_argument('--k',  type=int, default=10)
    args = parser.parse_args()

    if args.model == "c3f2":
        model = small.c3f2()
    elif args.model == "fconv":
        model = small.fconv()
    elif args.model == "mlp":
        model = small.MLP()
    elif args.model == "wideresnet":
        model = medium.WideResNet(args.N, args.k)
    serializers.load_npz(args.resume, model)

    def plot_color_filter(out_fn, weight):
        fig = plt.figure(figsize=(16, 16))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        nb_grids = int(np.ceil(np.sqrt(len(weight))))
        for idx in range(len(weight)):
            w = np.transpose(weight[idx], (1, 2, 0)) * 255
            ax = fig.add_subplot(nb_grids, nb_grids, idx + 1, xticks=[], yticks=[])
            ax.imshow(w, cmap=plt.cm.gray, interpolation='nearest')

        fig.savefig(out_fn)

    def plot_filter(out_fn, weight):
        fig = plt.figure(figsize=(16, 16))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        nb_grids = int(np.ceil(np.sqrt(np.prod(weight.shape[:2]))))
        cnt = 0
        for i in range(weight.shape[0]):
            for j in range(weight.shape[1]):
                w = weight[i, j] * 255
                ax = fig.add_subplot(nb_grids, nb_grids, cnt + 1, xticks=[], yticks=[])
                ax.imshow(w, cmap=plt.cm.gray, interpolation='nearest')
                cnt += 1

        fig.savefig(out_fn)

    conv0 = model.conv0.W.data
    plot_color_filter("filter/conv0.png", model.conv0.W.data)

    for n in range(1, args.N, 1):
        plot_filter("filter/block1_{}_1.png".format(n), getattr(model, "block1_{}".format(n)).conv1.W.data)
        plot_filter("filter/block1_{}_2.png".format(n), getattr(model, "block1_{}".format(n)).conv2.W.data)
        plot_filter("filter/block2_{}_1.png".format(n), getattr(model, "block2_{}".format(n)).conv1.W.data)
        plot_filter("filter/block2_{}_2.png".format(n), getattr(model, "block2_{}".format(n)).conv2.W.data)
        plot_filter("filter/block3_{}_1.png".format(n), getattr(model, "block3_{}".format(n)).conv1.W.data)
        plot_filter("filter/block3_{}_2.png".format(n), getattr(model, "block3_{}".format(n)).conv2.W.data)

    #import ipdb;ipdb.set_trace()
    """
    if hasattr(model, "conv1"):
        conv1 = model.conv1.W.data  # (out, in, H, W)
    else:
        conv1 = model.conv0.W.data

    for idx in range(len(conv1)):
        print(np.max(conv1[idx]), np.min(conv1[idx]))

    plt.figure()
    fig = plt.figure(figsize=(16, 9))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for idx in range(len(conv1)):
        weight = np.transpose(conv1[idx], (1, 2, 0)) * 255
        ax = fig.add_subplot(6, 6, idx + 1, xticks=[], yticks=[])
        ax.imshow(weight, cmap=plt.cm.gray, interpolation='nearest')

    fig.savefig("filter.png")
    """


if __name__ == '__main__':
    main()
