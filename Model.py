import torch
from torch import nn


def block(in_channel, out_channel):
    result = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True),
    )
    return result


# build model
class AIST_model(nn.Module):
    def __init__(self):
        super(AIST_model, self).__init__()
        self.feature = nn.Sequential(
            block(7, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Sequential(
            nn.Linear(512, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(self.avgpool(x), 1)
        output = self.fc2(self.fc1(x))
        return output
