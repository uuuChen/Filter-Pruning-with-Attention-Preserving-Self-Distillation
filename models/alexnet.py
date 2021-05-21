from typing import Any

import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'alexnet'
]


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv2_bn = nn.BatchNorm2d(192)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1_bn(self.conv1(x)), inplace=True), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2_bn(self.conv2(x)), inplace=True), kernel_size=2)
        x = F.relu(self.conv3_bn(self.conv3(x)), inplace=True)
        x = F.relu(self.conv4_bn(self.conv4(x)), inplace=True)
        x = F.max_pool2d(F.relu(self.conv5_bn(self.conv5(x)), inplace=True), kernel_size=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x


def alexnet(**kwargs: Any) -> AlexNet:
    model = AlexNet(**kwargs)
    return model
