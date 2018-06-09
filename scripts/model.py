import torch
from torch import nn
from torch.autograd import Variable


def contracting_conv(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=(4, 4),
                     stride=2,
                     padding=1,
                     bias=False)


def expanding_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels,
                              kernel_size=(4, 4),
                              stride=2,
                              padding=1,
                              bias=False)


class ContractingBlock(nn.Module):
    def __init__(self, in_channels, negative_slope=0.2):
        super().__init__()
        out_channels = in_channels * 2
        self.hidden = nn.Sequential(
            contracting_conv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope)
        )

    def forward(self, x):
        return self.hidden(x)


class ExpandingBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = in_channels // 2
        self.hidden = nn.Sequential(
            expanding_conv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.hidden(x)


class Discriminator(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            contracting_conv(3, 128),
            nn.LeakyReLU(negative_slope)
        )
        self.hidden = nn.Sequential(
            ContractingBlock(128),
            ContractingBlock(256),
            ContractingBlock(512)
        )
        self.out = nn.Sequential(
            nn.Linear(1024*4*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.hidden(x)
        x = x.view(-1, 1024*4*4)
        x = self.out(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 1024*4*4)
        self.hidden = nn.Sequential(
            ExpandingBlock(1024),
            ExpandingBlock(512),
            ExpandingBlock(256)
        )
        self.out = nn.Sequential(
            expanding_conv(128, 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], 1024, 4, 4)
        x = self.hidden(x)
        x = self.out(x)
        return x


def noise(size, n_features=100):
    eps = Variable(torch.randn(size, n_features))
    if torch.cuda.is_available():
        return eps.cuda()
    return eps


def init_weights(obj):
    classname = obj.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        obj.weight.data.normal_(.0, .002)