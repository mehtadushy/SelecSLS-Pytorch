#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Script for timing models in eval mode and torchscript eval modes.
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



def opts_parser():
    usage = 'Pass the model and'
    parser = ArgumentParser(description=usage)
    parser.add_argument(
        '--num_iter', type=int, default=50,
        help='Number of iterations to average over.')
    parser.add_argument(
        '--model_class', type=str, default='selecsls', metavar='FILE',
        help='Select model type to use (DenseNet, SelecSLS, ResNet etc.)')
    parser.add_argument(
        '--model_config', type=str, default='SelecSLS84', metavar='NET_CONFIG',
        help='Select the model configuration')
    parser.add_argument(
        '--model_weights', type=str, default='./weights/SelecSLS84_statedict.pth', metavar='FILE',
        help='Path to model weights')
    parser.add_argument(
        '--input_size', type=int, default=400,
        help='Input image size.')
    parser.add_argument(
        '--gpu_id', type=int, default=0,
        help='Which GPU to use.')
    parser.add_argument(
        '--pruned_and_fused', type=bool, default=False,
        help='Whether to zero out features with gamma below a certain threshold')
    parser.add_argument(
        '--gamma_thresh', type=float, default=1e-3,
        help='gamma threshold to use for simulating pruning. Set this to -1 to only fuse BN without pruning')
    return parser


def measure_cpu(model, x):
    # synchronize gpu time and measure fp
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        y_pred = model(x)
        elapsed_fp_nograd = time.time()-t0
    return elapsed_fp_nograd

def measure_gpu(model, x):
    # synchronize gpu time and measure fp
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        y_pred = model(x)
        torch.cuda.synchronize()
        elapsed_fp_nograd = time.time()-t0
    return elapsed_fp_nograd


def benchmark(model_class, model_config, gpu_id, num_iter, model_weights, input_size, pruned_and_fused, gamma_thresh):
    # Import the model module
    model_module = importlib.import_module('models.'+model_class)
    net = model_module.Net(nClasses=1000, config=model_config)
    net.load_state_dict(torch.load(model_weights, map_location= lambda storage, loc: storage))

    if pruned_and_fused:
        print('Pruning and fusing the model')
        net.prune_and_fuse(gamma_thresh, True)

    device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    print('\nEvaluating on GPU {}'.format(device))

    print('\nGPU, Batch Size: 1')
    x = torch.randn(1, 3, input_size, input_size)
    #Warm up
    for i in range(10):
      _ = measure_gpu(net, x.to(device))
    fp = []
    for i in range(num_iter):
        t  = measure_gpu(net, x.to(device))
        fp.append(t)
    print('Model FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')

    jit_net = torch.jit.trace(net, x.to(device))
    for i in range(10):
        _ = measure_gpu(jit_net, x.to(device))
    fp = []
    for i in range(num_iter):
        t  = measure_gpu(jit_net, x.to(device))
        fp.append(t)
    print('JIT FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')


    print('\nGPU, Batch Size: 16')
    x = torch.randn(16, 3, input_size, input_size)
    #Warm up
    for i in range(10):
        _ = measure_gpu(net, x.to(device))
    fp = []
    for i in range(num_iter):
        t  = measure_gpu(net, x.to(device))
        fp.append(t)
    print('Model FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')

    jit_net = torch.jit.trace(net, x.to(device))
    for i in range(10):
        _ = measure_gpu(jit_net, x.to(device))
    fp = []
    for i in range(num_iter):
        t  = measure_gpu(jit_net, x.to(device))
        fp.append(t)
    print('JIT FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')

    device = torch.device("cpu")
    print('\nEvaluating on {}'.format(device))
    net = net.to(device)

    print('\nCPU, Batch Size: 1')
    x = torch.randn(1, 3, input_size, input_size)
    #Warm up
    for i in range(10):
        _ = measure_cpu(net, x.to(device))
    fp = []
    for i in range(num_iter):
        t  = measure_cpu(net, x.to(device))
        fp.append(t)
    print('Model FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')

    jit_net = torch.jit.trace(net, x.to(device))
    for i in range(10):
        _ = measure_cpu(jit_net, x.to(device))
    fp = []
    for i in range(num_iter):
        t  = measure_cpu(jit_net, x.to(device))
        fp.append(t)
    print('JIT FP: '+str(np.mean(np.asarray(fp)*1000))+'ms')



def main():
    # parse command line
    torch.manual_seed(1234)
    parser = opts_parser()
    args = parser.parse_args()

    # run
    benchmark(**vars(args))

if __name__ == '__main__':
    main()
