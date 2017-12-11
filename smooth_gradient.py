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
import matplotlib.pyplot as plt
from skimage import color

import chainer
from chainer import cuda, Variable
from chainer.datasets import get_cifar10

from utils.get_model import get_model

from mllogger import MLLogger
logger = MLLogger(init=False)

class_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def clip_image(img, percentile=99):
    vmax = np.percentile(img, percentile)
    vmin = np.min(img)
    return np.clip((img - vmin) / (vmax - vmin), 0, 1)


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('resume')
    parser.add_argument('--nb_trials', type=int, default=50)
    parser.add_argument('--model', default='c5')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--nb_valid', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    start = time.time()
    logger.initialize("grad_"+args.model)
    logger.info(vars(args))
    np.random.seed(args.seed)
    save_dir = logger.get_savedir()
    logger.info("Written to {}".format(save_dir))
    logger.info('GPU: {}'.format(args.gpu))
    train_all, test = get_cifar10()

    if args.debug:
        valid = train_all[200:400]
    else:
        valid_choice = np.random.choice(range(len(train_all)),
                                        args.nb_valid, replace=False)
        valid = [x for idx, x in enumerate(train_all) if idx in valid_choice]

    print(len(valid))

    model = get_model(args.model, args.gpu, args.resume)

    # Get one image per iteration
    valid_iter = chainer.iterators.SerialIterator(
        valid, 1, repeat=False, shuffle=False)

    if not os.path.exists("grads"):
        os.makedirs("grads")

    chainer.config.train = False
    chainer.config.enable_backprop = True
    for idx, tup in enumerate(valid_iter):
        print(idx)
        img = tup[0][0]
        # Tile image to calculate all the trials at once
        inp = np.tile(img.copy()[np.newaxis, ...], (args.nb_trials, 1, 1, 1))
        label = tup[0][1][np.newaxis, ...]
        sigma = (inp.max() - inp.min()) * 0.025  # noise level
        model.cleargrads()
        inp = inp + np.random.randn(*inp.shape).astype(np.float32) * sigma  # Add noise to every image
        x = Variable(cuda.to_gpu(inp, args.gpu))
        xp = cuda.get_array_module(x)
        pred = model.get_feature(x, False)
        # print(class_list[int(cuda.to_cpu(xp.argmax(pred.data)))], class_list[int(label)])
        pred.grad = xp.ones(pred.shape, dtype=np.float32)
        pred.backward()
        mean_grad = cuda.to_cpu(xp.mean(x.grad.copy(), axis=0))
        mean_grad = np.max(np.abs(mean_grad), axis=0)
        mean_grad = color.gray2rgb(mean_grad)
        mean_grad = clip_image(mean_grad)
        orig_img = np.transpose(img, (1, 2, 0))
        masked = orig_img * mean_grad
        out = np.concatenate((mean_grad, masked, orig_img), axis=1)
        plt.imsave("grads/{:05d}.png".format(idx), out)
        model.cleargrads()

    print(time.time()-start)


if __name__ == '__main__':
    main()
