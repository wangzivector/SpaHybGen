# Inherent from [VGN](https://github.com/ethz-asl/vgn)

from builtins import super

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)


def conv_stride(in_channels, out_channels, kernel_size):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2
    )


def count_num_trainable_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


class Encoder(nn.Module):
    def __init__(self, in_channels, filters, kernels):
        super().__init__()
        self.conv1 = conv_stride(in_channels, filters[0], kernels[0])
        self.conv2 = conv_stride(filters[0], filters[1], kernels[1])
        self.conv3 = conv_stride(filters[1], filters[2], kernels[2])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, filters, kernels, voxel_size):
        super().__init__()
        self.voxel_size = voxel_size
        self.conv1 = conv(in_channels, filters[0], kernels[0])
        self.conv2 = conv(filters[0], filters[1], kernels[1])
        self.conv3 = conv(filters[1], filters[2], kernels[2])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = F.interpolate(x, self.voxel_size//4)

        x = self.relu(self.conv2(x))
        x = F.interpolate(x, self.voxel_size//2)

        x = self.relu(self.conv3(x))
        x = F.interpolate(x, self.voxel_size)
        return x


class VGN(nn.Module):
    def __init__(self, voxel_discreteness=80, orientation='quat', augment=False):
        super().__init__()
        self.voxel_discreteness = voxel_discreteness
        self.orientation = orientation
        filters_en, filters_de = [16, 32, 64], [64, 32, 16]
        if augment: filters_en, filters_de = [32, 64, 128], [128, 64, 32]
        self.encoder = Encoder(1, filters_en, [5, 3, 3])
        self.decoder = Decoder(filters_de[0], filters_de, [3, 3, 5], self.voxel_discreteness)
        self.conv_score = conv(filters_de[-1], 1, 5)
        self.conv_wren = conv(filters_de[-1], 1, 5)

        if self.orientation == "quat": rot_head_size = 4
        elif self.orientation == "so3": rot_head_size = 3
        elif self.orientation == "R6d": rot_head_size = 6
        self.conv_rot = conv(filters_de[-1], rot_head_size, 5)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        out_score = torch.sigmoid(self.conv_score(x))
        out_rot = F.normalize(self.conv_rot(x), dim=1) if self.orientation == "quat" else self.conv_rot(x)
        out_wren = torch.sigmoid(self.conv_wren(x))
        return out_score, out_rot, out_wren
