import torch.nn as nn
import torch.nn.functional as F


class BasicPreActResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicPreActResBlock, self).__init__()


        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)


        self.resConnect = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(x), inplace=True)
        y = self.conv1(x)

        y = F.relu(self.bn2(y), inplace=True)
        y = self.conv2(y)

        y += self.resConnect(x)
        return y


