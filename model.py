import torch
from torch import nn
from torchvision import models


class SnakeDetector(nn.Module):
    def __init__(self):
        super(SnakeDetector, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        for param in self.features.parameters():
            param.requires_grad = False
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 128),nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(128, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

