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
    def __init__(self, inp, skip, k, oup, isFirst, stride, groupLevel=0):
        super(SelecSLSBlock, self).__init__()
        self.stride = stride
        self.isFirst = isFirst
        assert stride in [1, 2]
        assert groupLevel in [0,1,2]  #Group level 0 is no grouping, 1 is some grouping, and 2 is (almost)depthwise

        #Process input with 4 conv blocks with the same number of input and output channels
        #print('Conv1 groups:'+str(2**max(math.floor(math.log2(k/64)),0))+' k:'+str(k))
        self.conv1 = nn.Sequential(
                nn.Conv2d(inp, k, 3, stride, 1,groups= 1 if groupLevel == 0 else int(2**max(math.floor(math.log2(k/64)),0)) if groupLevel == 1 else gcd(inp,k), bias=False, dilation=1),
                nn.BatchNorm2d(k),
                nn.ReLU(inplace=True)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(k, k, 1, 1, 0,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k),
                nn.ReLU(inplace=True)
                )
        #print('Conv3 groups:'+str(2**max(math.floor(math.log2(k/128)),0))+' k:'+str(k))
        self.conv3 = nn.Sequential(
                nn.Conv2d(k, k//2, 3, 1, 1,groups= 1 if groupLevel == 0 else int(2**max(math.floor(math.log2(k/128)),0)) if groupLevel == 1 else k//2, bias=False, dilation=1),
                nn.BatchNorm2d(k//2),
                nn.ReLU(inplace=True)
                )
        self.conv4 = nn.Sequential(
                nn.Conv2d(k//2, k, 1, 1, 0,groups= 1, bias=False, dilation=1),
                nn.BatchNorm2d(k),
                nn.ReLU(inplace=True)
                )
        #print('Conv5 groups:'+str(2**max(math.floor(math.log2(k/128)),0))+' k:'+str(k))
        self.conv5 = nn.Sequential(
                nn.Conv2d(k, k//2, 3, 1, 1,groups= 1 if groupLevel == 0 else int(2**max(math.floor(math.log2(k/128)),0)) if groupLevel == 1 else k//2, bias=False, dilation=1),
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
    def __init__(self, inp, k, oup, groupLevel=1, stride=2):
        super(SelecSLSHead, self).__init__()
        self.sls_head =  SelecSLSBlock(inp, 0, k, oup, True, stride, groupLevel)

    def forward(self, x):
        return self.sls_head([x])[0]

class Net(nn.Module):
    def __init__(self, nClasses=1000, netConfig='base', groupLevel = 0):
        super(Net, self).__init__()

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

        self.stem = conv_bn(3, 32, 2)
        self.features = []
        # building dlnas core 
        for inp, skip, k, oup, isFirst, stride  in self.selecSLS_config:
            self.features.append(SelecSLSBlock(inp, skip, k, oup, isFirst, stride, groupLevel))
        self.features = nn.Sequential(*self.features)

        if netConfig=='SelecSLS_Final':
            print('SelecSLS_Final')
            self.head = nn.Sequential(
                    conv_1x1_bn(416, 512),
                    conv_bn(512, 512, 1),
                    conv_bn(512, 1024, 2),
                    conv_1x1_bn(1024, 1280),
                    )
        elif netConfig=='SelecSLS_Smallhead':
            print('SelecSLS_Smallhead')
            self.head = nn.Sequential(
                    conv_bn(416, 512, 2),
                    conv_1x1_bn(512, 1280)
                    )
        elif netConfig=='SelecSLS_Evenbiggerhead':
            print('SelecSLS_Evenbiggerhead')
            self.head = nn.Sequential(
                    SelecSLSHead(416, 512, 1280, groupLevel)
                    )
        elif netConfig=='SelecSLS_Biggerhead':
            print('SelecSLS_Biggerhead')
            self.head = nn.Sequential(
                    conv_bn(416, 756, 2),
                    conv_bn(756, 1024, 1),
                    conv_bn(1024, 1024, 2),
                    conv_1x1_bn(1024, 1280),
                    )
        elif netConfig=='SelecSLS_Mixedhead':
            print('SelecSLS_Mixedhead')
            self.head = nn.Sequential(
                    SelecSLSHead(416, 640, 1280, groupLevel),
                    conv_bn(1280, 1280, 1),
                    conv_bn(1280, 1280, 2),
                    conv_1x1_bn(1280, 1280),
                    )
        elif netConfig=='SelecSLS_Plainhead':
            print('SelecSLS_Plainhead')
            self.head = nn.Sequential(
                    conv_bn(416, 1024, 2),
                    conv_1x1_bn(1024, 1024),
                    conv_bn(1024, 1024, 1),
                    conv_1x1_bn(1024, 1280),
                    conv_bn(1280, 1280, 2),
                    conv_1x1_bn(1280, 1280),
                    )
        elif netConfig=='SelecSLS_Expressivehead':
            print('SelecSLS_Expressivehead')
            self.head = nn.Sequential(
                    conv_bn(416, 756, 1),
                    nn.MaxPool2d(3,2,1),
                    SelecSLSHead(756, 756, 1024, groupLevel, stride=1),
                    conv_bn(1024, 1280, 1),
                    )
        elif netConfig=='SelecSLS_Althead':
            print('SelecSLS_Althead')
            self.head = nn.Sequential(
                    conv_bn(416, 756, 1),
                    nn.MaxPool2d(4,2,1),
                    conv_bn(756, 960, 1),
                    conv_1x1_bn(960, 1280)
                    )
        elif netConfig=='SelecSLS_Althead2':
            print('SelecSLS_Althead2')
            self.head = nn.Sequential(
                    conv_bn(416, 756, 1),
                    nn.MaxPool2d(4,2,1),
                    conv_bn(756, 960, 1),
                    nn.AdaptiveAvgPool2d(output_size=(1,1)),
                    conv_1x1_bn(960, 1280)
                    )
        elif netConfig=='SelecSLS_Doublehead':
            print('SelecSLS_Doublehead')
            self.head = nn.Sequential(
                    SelecSLSHead(416, 512, 960, groupLevel),
                    SelecSLSHead(960, 1024, 1280, groupLevel),
                    )
        else:
            raise ValueError('Invalid net configuration '+netConfig+' !!!')

        # building classifier
        self.classifier = nn.Sequential(
                nn.Linear(1280, nClasses),
        )


    def forward(self, x):
        x = self.stem(x)
        x = self.features([x])
        x = self.head(x[0])
        x = x.mean(3).mean(2)
        #x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x)
        return x
