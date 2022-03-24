import numpy as np

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torchvision.models as models

model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class LocalizerAlexNet(nn.Module):
    def __init__(self, num_classes=20, pretrained=False):
        super(LocalizerAlexNet, self).__init__()

        # Refer to https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
        self.features = torch.hub.load('pytorch/vision:v0.10.0',
                                       'alexnet',
                                       pretrained=pretrained).features[: -1]

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 20, 1, stride=1),
        )


    def forward(self, x):
        #TODO: Define forward pass

        x = self.features(x)
        x = self.classifier(x)

        return x


class LocalizerAlexNetRobust(nn.Module):
    def __init__(self, num_classes=20, pretrained=False):
        super(LocalizerAlexNetRobust, self).__init__()
        #TODO: Define model

        # Refer to https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
        self.features = torch.hub.load('pytorch/vision:v0.10.0',
                                       'alexnet',
                                       pretrained=pretrained).features[: -1]

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 20, 1, stride=1),
        )


    def forward(self, x):
        #TODO: Define forward pass

        x = self.features(x)
        x = self.classifier(x)

        return x


def localizer_alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNet(pretrained=pretrained)
    #TODO: Initialize weights correctly based on whethet it is pretrained or not

    model.classifier.apply(initialize_weights)

    if not pretrained:
        model.features.apply(initialize_weights)

    return model


def localizer_alexnet_robust(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = LocalizerAlexNetRobust(pretrained=pretrained)
    #TODO: Initialize weights correctly based on whethet it is pretrained or not

    model.classifier.apply(initialize_weights)

    if not pretrained:
        model.features.apply(initialize_weights)

    return model

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)