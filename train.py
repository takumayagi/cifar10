#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from six.moves import range

import os
import time
import argparse

import numpy as np

import chainer
from chainer.dataset import convert
from chainer import serializers
from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

from utils.get_model import get_model

from mllogger import MLLogger
logger = MLLogger(init=False)


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--model', default='c3f2')
    parser.add_argument('--batchsize', '-b', type=int, default=64)
    parser.add_argument('--learnrate', '-l', type=float, default=0.05)
    parser.add_argument('--epoch', '-e', type=int, default=300)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--N',  type=int, default=9)
    parser.add_argument('--k',  type=int, default=10)
    parser.add_argument('--out', '-o', default='result')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', '-r', default='')
    args = parser.parse_args()
    start = time.time()
    logger.initialize("outputs_"+args.model)
    logger.info(vars(args))
    save_dir = logger.get_savedir()
    logger.info("Written to {}".format(save_dir))
    logger.info('GPU: {}'.format(args.gpu))
    logger.info('# Minibatch-size: {}'.format(args.batchsize))
    logger.info('# epoch: {}'.format(args.epoch))
    logger.info('')
    train, test = get_cifar10()
    if args.debug:
        train, test = train[:200], test[:200]
    train_count, test_count = len(train), len(test)

    model = get_model(args.model, args.gpu, args.resume)

    optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    sum_accuracy = 0
    sum_loss = 0
    st = time.time()
    iter_cnt = 0
    chainer.config.train = True
    chainer.config.enable_backprop = True
    while train_iter.epoch < args.epoch:
        batch = train_iter.next()
        if train_iter.epoch % 60 == 0 and train_iter.is_new_epoch:
            optimizer.lr *= 0.2
            logger.info('Reducing learning rate to: {}'.format(optimizer.lr))

        x_array, t_array = convert.concat_examples(batch, args.gpu)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array)
        model.cleargrads()
        loss = model(x, t)
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

        if train_iter.is_new_epoch:
            train_loss = sum_loss / train_count
            train_acc = sum_accuracy / train_count
            # evaluation
            sum_accuracy = 0
            sum_loss = 0
            chainer.config.train = False
            chainer.config.enable_backprop = False
            for batch in test_iter:
                x_array, t_array = convert.concat_examples(batch, args.gpu)
                x = chainer.Variable(x_array)
                t = chainer.Variable(t_array)
                loss = model(x, t)
                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(model.accuracy.data) * len(t.data)

            test_iter.reset()
            chainer.config.train = True
            chainer.config.enable_backprop = True
            test_loss = sum_loss / test_count
            test_acc = sum_accuracy / test_count
            message_str = "Epoch {}: train loss={:.4f}, acc={:.4f}, test loss={:.4f}, acc={:.4f}, elapsed={}"
            logger.info(message_str.format(train_iter.epoch, train_loss, train_acc,
                                     test_loss, test_acc, time.time()-st))
            st = time.time()
            sum_accuracy = 0
            sum_loss = 0
            if not args.debug:
                serializers.save_npz(os.path.join(save_dir, "model_ep_{}.npz".format(train_iter.epoch)), model)
        iter_cnt += 1


if __name__ == '__main__':
    main()
