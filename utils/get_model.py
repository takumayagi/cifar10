#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2017 Takuma Yagi <tyagi@iis.u-tokyo.ac.jp>
#
# Distributed under terms of the MIT license.

import sys
sys.path.append("../")

from chainer import cuda, serializers
from models import small
from models import medium


def get_model(model_name, gpu, resume):
    if model_name == "c3f2":
        model = small.c3f2()
    elif model_name == "fconv":
        model = small.fconv()
    elif model_name == "mlp":
        model = small.MLP()
    elif model_name == "wideresnet":
        model = medium.WideResNet(args.N, args.k)
    elif model_name == "c5":
        model = small.C5()
    else:
        print("Invalid model={}".format(model_name))
        exit(1)

    if resume != "":
        serializers.load_npz(resume, model)

    if gpu >= 0:
        cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    return model
