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
from chainer import cuda
from chainer.datasets import get_cifar10

from utils.get_model import get_model

from mllogger import MLLogger
logger = MLLogger(init=False)


def get_color_map_nipy(gradation_num):
    colors = []
    for idx in [x*255/gradation_num for x in xrange(gradation_num)]:
        colors.append(plt.cm.nipy_spectral(idx)[0:3])
    return (np.array(colors)[::-1,(2,1,0)]*255).astype(np.int)

def plot_error(data, label, predictions, size=32):
    colors = get_color_map_nipy(10)
    font = cv2.FONT_HERSHEY_SIMPLEX
    class_names = ["air", "aut", "bir", "cat", "der", "dog", "frg", "hor", "shp", "trk"]
    #import pdb;pdb.set_trace()
    error_idxs = [idx for idx, (pred, gt) in enumerate(zip(predictions, label)) if pred != gt]
    #error_idxs = error_idxs[:100]
    vert = size // 2 * 3
    nb_grids = int(np.ceil(np.sqrt(len(error_idxs))))
    canvas = np.ones((vert * nb_grids, size * nb_grids, 3), dtype=np.uint8) * 255

    for cnt, idx in enumerate(error_idxs):
        xidx = cnt % nb_grids
        yidx = cnt // nb_grids
        canvas[yidx*vert:yidx*vert+size, xidx*size:(xidx+1)*size] = \
            np.transpose(data[idx], (1, 2, 0))[..., ::-1] * 255
        cv2.rectangle(canvas, (xidx * size, yidx * vert),
                      (xidx * size - 1, yidx * vert + size - 1), colors[label[idx]], 1)
        cv2.putText(canvas, class_names[predictions[idx]],
                    (xidx * size, yidx * vert + vert - 4),
                    font, 0.5, (0, 0, 0), 1, 16)
    return canvas

def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--model', default='c3f2')
    parser.add_argument('--batchsize', '-b', type=int, default=64)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--N',  type=int, default=9)
    parser.add_argument('--k',  type=int, default=10)
    parser.add_argument('--nb_valid',  type=int, default=10000)
    parser.add_argument('--seed',  type=int, default=1701)
    parser.add_argument('--out', '-o', default='result')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', '-r', default='')
    args = parser.parse_args()
    start = time.time()
    logger.initialize("evals_"+args.model)
    logger.info(vars(args))
    np.random.seed(args.seed)
    save_dir = logger.get_savedir()
    logger.info("Written to {}".format(save_dir))
    logger.info('GPU: {}'.format(args.gpu))
    logger.info('# Minibatch-size: {}'.format(args.batchsize))
    logger.info('')
    train_all, test = get_cifar10()

    if args.debug:
        valid = train_all[200:400]
        batchsize = 20
    else:
        valid_choice = np.random.choice(range(len(train_all)),
                                        args.nb_valid, replace=False)
        valid = [x for idx, x in enumerate(train_all) if idx in valid_choice]
        batchsize = args.batchsize

    valid_cnt = len(valid)
    print(valid_cnt)

    model = get_model(args.model, args.gpu, args.resume)
    valid_iter = chainer.iterators.SerialIterator(
        valid, batchsize, repeat=False, shuffle=False)

    sum_accuracy = 0
    sum_loss = 0
    chainer.config.train = False
    chainer.config.enable_backprop = False
    predictions = []
    for idx, batch in enumerate(valid_iter):
        x_array, t_array = convert.concat_examples(batch, args.gpu)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array)
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)
        predictions.extend(np.argmax(cuda.to_cpu(model.pred.data), axis=1).tolist())

    valid_iter.reset()
    valid_loss = sum_loss / valid_cnt
    valid_acc = sum_accuracy / valid_cnt
    message_str = "Valid loss={:.4f}, acc={:.4f}"
    logger.info(message_str.format(valid_loss, valid_acc))

    canvas = plot_error([x[0] for x in valid], [x[1] for x in valid], predictions)
    cv2.imwrite("error.jpg", canvas)
    print(time.time()-start)


if __name__ == '__main__':
    main()
