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
    parser.add_argument('--nb_valid',  type=int, default=10000)
    parser.add_argument('--seed',  type=int, default=1701)
    parser.add_argument('--out', '-o', default='result')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', '-r', default='')
    args = parser.parse_args()
    start = time.time()
    logger.initialize("outputs_"+args.model)
    logger.info(vars(args))
    np.random.seed(args.seed)
    save_dir = logger.get_savedir()
    logger.info("Written to {}".format(save_dir))
    logger.info('GPU: {}'.format(args.gpu))
    logger.info('# Minibatch-size: {}'.format(args.batchsize))
    logger.info('# epoch: {}'.format(args.epoch))
    logger.info('')
    train_all, test = get_cifar10()

    if args.debug:
        train, valid, test = train_all[:200], train_all[200:400], test[:200]
        batchsize = 20
    else:
        valid_choice = np.random.choice(range(len(train_all)),
                                        args.nb_valid, replace=False)
        train = [x for idx, x in enumerate(train_all) if idx not in valid_choice]
        valid = [x for idx, x in enumerate(train_all) if idx in valid_choice]
        batchsize = args.batchsize
        #import pdb;pdb.set_trace()

    train_cnt, valid_cnt, test_cnt = len(train), len(valid), len(test)
    print(train_cnt, valid_cnt, test_cnt)

    model = get_model(args.model, args.gpu, args.resume)
    optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, batchsize,
                                                 repeat=False, shuffle=False)

    sum_accuracy = 0
    sum_loss = 0
    st = time.time()
    iter_cnt = 0
    chainer.config.train = True
    chainer.config.enable_backprop = True
    logger.info("Training...")
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
            train_loss = sum_loss / train_cnt
            train_acc = sum_accuracy / train_cnt

            # validation
            sum_accuracy = 0
            sum_loss = 0
            chainer.config.train = False
            chainer.config.enable_backprop = False
            for batch in valid_iter:
                x_array, t_array = convert.concat_examples(batch, args.gpu)
                x = chainer.Variable(x_array)
                t = chainer.Variable(t_array)
                loss = model(x, t)
                sum_loss += float(loss.data) * len(t.data)
                sum_accuracy += float(model.accuracy.data) * len(t.data)

            valid_iter.reset()
            valid_loss = sum_loss / valid_cnt
            valid_acc = sum_accuracy / valid_cnt
            message_str = "Epoch {}: train loss={:.4f}, acc={:.4f}, valid loss={:.4f}, acc={:.4f}, elapsed={}"
            logger.info(message_str.format(train_iter.epoch, train_loss, train_acc,
                                     valid_loss, valid_acc, time.time()-st))
            st = time.time()
            chainer.config.train = True
            chainer.config.enable_backprop = True
            sum_accuracy = 0
            sum_loss = 0
            if not args.debug:
                serializers.save_npz(os.path.join(
                    save_dir, "model_holdout_ep_{}.npz".format(train_iter.epoch)), model)
        iter_cnt += 1

    if not test:
        print(time.time()-start)
        exit(1)

    logger.info("Test...")
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)
    sum_accuracy = 0
    sum_loss = 0
    chainer.config.train = False
    chainer.config.enable_backprop = False
    st = time.time()
    for batch in test_iter:
        x_array, t_array = convert.concat_examples(batch, args.gpu)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array)
        loss = model(x, t)
        sum_loss += float(loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)
    test_loss = sum_loss / test_cnt
    test_acc = sum_accuracy / test_cnt
    message_str = "test loss={:.4f}, acc={:.4f}, elapsed={}"
    logger.info(message_str.format(test_loss, test_acc, time.time()-st))
    print(time.time()-start)

if __name__ == '__main__':
    main()
