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

import numpy as np
import cv2
from chainer.datasets import get_cifar10

# RGB
train, test = get_cifar10()

nb_rows = 24
canvas = np.zeros((32 * nb_rows, 32 * nb_rows, 3))
for x in range(nb_rows):
    for y in range(nb_rows):
        img = np.transpose((train[x+y*nb_rows][0]*255).astype(np.uint8), (1, 2, 0))
        canvas[y*32:(y+1)*32,x*32:(x+1)*32,:] = img[:, :, ::-1]

cv2.imwrite("test.jpg", canvas)
