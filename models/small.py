#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import chainer
import chainer.functions as F
import chainer.links as L


class c3f2(chainer.Chain):
    def __init__(self):
        super(c3f2, self).__init__(
            conv1=L.Convolution2D(3, 32, 5, stride=1, pad=2),
            conv2=L.Convolution2D(32, 32, 5, stride=1, pad=2),
            conv3=L.Convolution2D(32, 64, 5, stride=1, pad=2),
            fc4=L.Linear(1024, 512),
            fc5=L.Linear(512, 10),
        )

    def get_feature(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv3(h)), 2)  # 4x4x64
        h = F.dropout(F.relu(self.fc4(h)), ratio=0.5)
        return F.squeeze(h)

    def __call__(self, x, t, train=True):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 3, stride=2)
        h = F.max_pooling_2d(F.relu(self.conv3(h)), 2)  # 4x4x64
        h = F.dropout(F.relu(self.fc4(h)), ratio=0.5)
        h = self.fc5(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)

        if train:
            return self.loss
        else:
            self.pred = F.softmax(h)
            return self.pred


class fconv(chainer.Chain):
    def __init__(self):
        super(fconv, self).__init__(
            conv1=L.Convolution2D(3, 32, 5),
            conv2=L.Convolution2D(32, 64, 5),
            conv3=L.Convolution2D(64, 128, 3),
            conv4=L.Convolution2D(128, 256, 3),
            fc=L.Linear(256, 10),
        )

    def __call__(self, x, t, train=True):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = self.fc(h)

        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)

        if train:
            return self.loss
        else:
            self.pred = F.softmax(h)
            return self.pred


class MLP(chainer.Chain):
    def __init__(self):
        super(MLP, self).__init__(
            fc1=L.Linear(32*32*3, 1024),
            fc2=L.Linear(1024, 10)
        )

    def __call__(self, x, t, train=True):
        h = F.reshape(x, (len(x), -1))
        h = F.relu(self.fc1(h))
        h = self.fc2(h)
        self.loss = F.softmax_cross_entropy(h, t)
        self.accuracy = F.accuracy(h, t)

        return self.loss
