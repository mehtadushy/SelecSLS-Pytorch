'''
Pytorch implementation of SelecSLS Network architecture as described in
XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera.
The network architecture performs comparable to ResNet-50 while being 1.4-1.8x faster,
particularly with larger image sizes. The network architecture has a much smaller memory
footprint, and can be used as a drop in replacement for ResNet-50 in various tasks.
This Pytorch implementation establishes an official baseline of the model on ImageNet

Code Author: Dushyant Mehta (dmehta[at]mpi-inf.mpg.de)
'''
from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import fractions


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class SelecSLSBlock(nn.Module):
    def __init__(self, inp, skip, k, oup, isFirst, stride):
        super(SelecSLSBlock, self).__init__()
        self.stride = stride
        self.isFirst = isFirst
        assert stride in [1, 2]

        #Process input with 4 conv blocks with the same number of input and output channels
        self.conv1 = nn.Sequential(
                nn.Conv2d(inp, k, 3, stride, 1,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k),
                nn.ReLU(inplace=True)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(k, k, 1, 1, 0,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k),
                nn.ReLU(inplace=True)
                )
        self.conv3 = nn.Sequential(
                nn.Conv2d(k, k//2, 3, 1, 1,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k//2),
                nn.ReLU(inplace=True)
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(k//2, k, 1, 1, 0,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k),
                nn.ReLU(inplace=True)
                )
        self.conv5 = nn.Sequential(
                nn.Conv2d(k, k//2, 3, 1, 1,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k//2),
                nn.ReLU(inplace=True)
                )
        self.conv6 = nn.Sequential(
                nn.Conv2d(2*k + (0 if isFirst else skip), oup, 1, 1, 0,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        assert isinstance(x,list)
        assert len(x) in [1,2]

        d1 = self.conv1(x[0])
        d2 = self.conv3(self.conv2(d1))
        d3 = self.conv5(self.conv4(d2))
        if self.isFirst:
            out = self.conv6(torch.cat([d1, d2, d3], 1))
            return [out, out]
        else:
            return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)) , x[1]]

class SelecSLSHead(nn.Module):
    def __init__(self, inp, k, oup, stride=2):
        super(SelecSLSHead, self).__init__()
        self.sls_head =  SelecSLSBlock(inp, 0, k, oup, True, stride)

    def forward(self, x):
        return self.sls_head([x])[0]

class Net(nn.Module):
    def __init__(self, nClasses=1000, config='SelecSLS60'):
        super(Net, self).__init__()

        #Stem
        self.stem = conv_bn(3, 32, 2)

        #Core Network
        self.features = []
        if config=='SelecSLS60':
            print('SelecSLS60')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 32,   0,  64,  64,  True,  2],
                [ 64,  64,  64, 128,  False, 1],
                [128,   0, 128, 128,  True,  2],
                [128, 128, 128, 128,  False, 1],
                [128, 128, 128, 288,  False, 1],
                [288,   0, 288, 288,  True,  2],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 288,  False, 1],
                [288, 288, 288, 416,  False, 1],
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(416, 756, 2),
                    conv_bn(756, 1024, 1),
                    conv_bn(1024, 1024, 2),
                    conv_1x1_bn(1024, 1280),
                    )
        elif config=='SelecSLS78':
            print('SelecSLS78')
            #Define configuration of the network after the initial neck
            self.selecSLS_config = [
                #inp,skip, k, oup, isFirst, stride
                [ 32,   0,  72,  72,  True,  2],
                [ 72,  72,  72,  72,  False, 1],
                [ 72,  72,  72, 144,  False, 1],
                [144,   0, 144, 144,  True,  2],
                [144, 144, 144, 144,  False, 1],
                [144, 144, 144, 144,  False, 1],
                [144, 144, 144, 320,  False, 1],
                [320,   0, 320, 320,  True,  2],
                [320, 320, 320, 320,  False, 1],
                [320, 320, 320, 320,  False, 1],
                [320, 320, 320, 320,  False, 1],
                [320, 320, 320, 512,  False, 1],
            ]
            #Head can be replaced with alternative configurations depending on the problem
            self.head = nn.Sequential(
                    conv_bn(512, 960, 2),
                    conv_bn(960, 1024, 1),
                    conv_bn(1024, 1024, 2),
                    conv_1x1_bn(1024, 1280),
                    )
        else:
            raise ValueError('Invalid net configuration '+config+' !!!')

        #Build SelecSLS Core 
        for inp, skip, k, oup, isFirst, stride  in self.selecSLS_config:
            self.features.append(SelecSLSBlock(inp, skip, k, oup, isFirst, stride))
        self.features = nn.Sequential(*self.features)

        #Classifier To Produce Inputs to Softmax
        self.classifier = nn.Sequential(
                nn.Linear(1280, nClasses),
        )


    def forward(self, x):
        x = self.stem(x)
        x = self.features([x])
        x = self.head(x[0])
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        #x = F.log_softmax(x)
        return x
