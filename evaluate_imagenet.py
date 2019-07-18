#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Script for evaluating accuracy on Imagenet Validation Set.
'''
import os
import logging
import sys
import time
from argparse import ArgumentParser
import importlib

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
from util.imagenet_data_loader import get_data_loader



def opts_parser():
    usage = 'Configure the dataset using imagenet_data_loader'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '--model_class', type=str, default='selecsls', metavar='FILE',
        help='Select model type to use (DenseNet, SelecSLS, ResNet etc.)')
    parser.add_argument(
        '--model_config', type=str, default='SelecSLS60', metavar='NET_CONFIG',
        help='Select the model configuration')
    parser.add_argument(
        '--model_weights', type=str, default='./weights/SelecSLS60_statedict.pth', metavar='FILE',
        help='Path to model weights')
    parser.add_argument(
        '--imagenet_base_path', type=str, default='/HPS/TwoHandCNN/nobackup/ImageNet/', metavar='FILE',
        help='Path to ImageNet dataset')
    parser.add_argument(
        '--gpu_id', type=int, default=0,
        help='Which GPU to use.')
    return parser


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def evaluate_imagenet_validation_accuracy(model_class, model_config, model_weights, imagenet_base_path, gpu_id):
    model_module = importlib.import_module('models.'+model_class)
    net = model_module.Net(nClasses=1000, config=model_config)
    net.load_state_dict(torch.load(model_weights, map_location= lambda storage, loc: storage))

    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()

    _,test_loader = get_data_loader(augment=False, batch_size=100, base_path=imagenet_base_path)
    with torch.no_grad():
        val1_err = []
        val5_err = []
        for x, y in test_loader:
            pred = F.log_softmax(net(x.to(device)))
            top1, top5 = accuracy(pred, y.to(device), topk=(1, 5))
            val1_err.append(100-top1)
            val5_err.append(100-top5)
        avg1_err=  float(np.sum(val1_err)) / len(val1_err)
        avg5_err=  float(np.sum(val5_err)) / len(val5_err)
    print('Top-1 Error: {} Top-5 Error {}'.format(avg1_err, avg5_err))


def main():
    # parse command line
    torch.manual_seed(1234)
    parser = opts_parser()
    args = parser.parse_args()

    # run
    evaluate_imagenet_validation_accuracy(**vars(args))

if __name__ == '__main__':
    main()
