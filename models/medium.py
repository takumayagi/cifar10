#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import numpy as np

import chainer
from chainer import initializers
from chainer import Variable
import chainer.functions as F
import chainer.links as L


class BlockA(chainer.Chain):
    def __init__(self, in_size, out_size, stride=2):
        super(BlockA, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.bn1=L.BatchNormalization(in_size)
            self.conv1=L.Convolution2D(in_size, out_size, 3, pad=1, initialW=initialW, nobias=True)
            self.bn2=L.BatchNormalization(out_size)
            self.conv2=L.Convolution2D(out_size, out_size, 3, pad=1, stride=stride, initialW=initialW, nobias=True)
            self.conv_skip=L.Convolution2D(in_size, out_size, 1, stride=stride, initialW=initialW, nobias=True)

    def __call__(self, x):
        # Full pre-activation
        h = self.conv1(F.relu(self.bn1(x)))
        h = F.dropout(h, ratio=0.5)
        h = self.conv2(F.relu(self.bn2(h)))
        return h + self.conv_skip(x)


class BlockB(chainer.Chain):
    def __init__(self, in_size):
        super(BlockB, self).__init__()
        initialW = initializers.HeNormal()

        with self.init_scope():
            self.bn1=L.BatchNormalization(in_size)
            self.conv1=L.Convolution2D(in_size, in_size, 3, pad=1, initialW=initialW, nobias=True)
            self.bn2=L.BatchNormalization(in_size)
            self.conv2=L.Convolution2D(in_size, in_size, 3, pad=1, initialW=initialW, nobias=True)

    def __call__(self, x):
        # Full pre-activation
        h = self.conv1(F.relu(self.bn1(x)))
        h = F.dropout(h, ratio=0.5)
        h = self.conv2(F.relu(self.bn2(h)))
        return h + x


class WideResNet(chainer.Chain):
    def __init__(self, N=9, k=10):
        super(WideResNet, self).__init__()
        self.N = N
        self.k = k
        with self.init_scope():
            self.conv0=L.Convolution2D(3, 16, 3, pad=1)
            self.bn0=L.BatchNormalization(16)
            self.block1_1 = BlockA(16, 16*k, 1)
            for n in range(2, N+1, 1):
                setattr(self, "block1_{}".format(n), BlockB(16*k))
            self.block2_1 = BlockA(16*k, 32*k)
            for n in range(2, N+1, 1):
                setattr(self, "block2_{}".format(n), BlockB(32*k))
            self.block3_1 = BlockA(32*k, 64*k)
            for n in range(2, N+1, 1):
                setattr(self, "block3_{}".format(n), BlockB(64*k))
            self.bn=L.BatchNormalization(64*k)
            self.fc=L.Linear(64*k, 1000)

    def get_feature(self, x):
        h = F.relu(self.bn0(self.conv0(x)))
        for n in range(1, self.N+1, 1):
            h = getattr(self, "block1_{}".format(n))(h)
        for n in range(1, self.N+1, 1):
            h = getattr(self, "block2_{}".format(n))(h)
        for n in range(1, self.N+1, 1):
            h = getattr(self, "block3_{}".format(n))(h)
        return F.squeeze(F.average_pooling_2d(F.relu(self.bn(h)), 8, 1))

    def __call__(self, x, t):
        h = F.relu(self.bn0(self.conv0(x)))
        for n in range(1, self.N+1, 1):
            h = getattr(self, "block1_{}".format(n))(h)
        for n in range(1, self.N+1, 1):
            h = getattr(self, "block2_{}".format(n))(h)
        for n in range(1, self.N+1, 1):
            h = getattr(self, "block3_{}".format(n))(h)
        h = F.average_pooling_2d(F.relu(self.bn(h)), 8, 1)
        h = self.fc(h)
        if t is not None:
            self.loss = F.softmax_cross_entropy(h, t)
            self.accuracy = F.accuracy(h, t)
            return self.loss
