from typing import Any

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=4, padding=5)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=(5, 5), padding=2)
        self.conv2_bn = nn.BatchNorm2d(192)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=(3, 3), padding=1)
        self.conv3_bn = nn.BatchNorm2d(384)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.conv5_bn = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2))
        out = self.conv3(out)
        out = self.conv3_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv4(out)
        out = self.conv4_bn(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.3, training=self.training)
        out = self.conv5(out)
        out = self.conv5_bn(out)
        out = F.relu(out)
        out = F.max_pool2d(out, kernel_size=(2, 2))

        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out


def alexnet(**kwargs: Any) -> AlexNet:
    model = AlexNet(**kwargs)
    return model
