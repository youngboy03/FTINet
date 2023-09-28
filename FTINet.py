# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from models.ECM import ECM
from AFLM.AFLM import AFLM


from cupy_layers.aggregation_zeropad import LocalConvolution

class CIformer(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CIformer, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size

        self.key = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size // 2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(inplace=True)
        )
        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2 * dim, dim // factor, 1, bias=False),
            nn.BatchNorm2d(dim // factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // factor, pow(kernel_size, 2) * dim // share_planes, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=pow(kernel_size, 2) * dim // share_planes)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.conv1x2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        self.local_conv = LocalConvolution(dim, dim, kernel_size=self.kernel_size, stride=1,
                                           padding=(self.kernel_size - 1) // 2, dilation=1)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv2d(dim, attn_chs, 1),
            nn.BatchNorm2d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_chs, self.radix * dim, 1)
        )

    def forward(self, x):  # 2.64.17.17

        k = self.key(x)  # 2.64.17.17
        qk = torch.cat([x, k], dim=1)  # 2.128.17.17
        b, c, qkh, qkw = qk.size()
        A = self.embed(qk)  # 2.200.17.17
        A = A.view(b, 1, -1, self.kernel_size * self.kernel_size, qkh, qkw)  # 2.1.8.25.17.17
        x = self.conv1x1(x)  # 2.64.17.17
        x = self.local_conv(x, A)  # 2.64.17.17
        x = self.bn(x)
        x = self.act(x)
        B, C, H, W = x.shape
        x = x.view(B, C, 1, H, W)  # 2.64.1.17.17
        k = k.view(B, C, 1, H, W)  # 2.64.1.17.17
        x = torch.cat([x, k], dim=2)  # 2.64.2.17.17

        x_ = x.sum(dim=2)  # 2.64.17.17

        x=self.conv1x2(x_)


        return x.contiguous()



class DropBlock2D(nn.Module):
    #
    def __init__(self, drop_prob, block_size):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob  # 0.1
        self.block_size = block_size  # 3

    def forward(self, x):  # 256.128.5.5
        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            gamma = self._compute_gamma(x)
            # sample mask
            # 256.5.5    torch.rand返回服从均匀分布的初始化后的tenosr
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()
            # x.device=cuda:0
            mask = mask.to(x.device)  # 256.5.5
            block_mask = self._compute_block_mask(mask)  # 256.5.5
            out = x * block_mask[:, None, :, :]
            # a=block_mask.numel()#numel()函数：返回数组中元素的个数
            # b=block_mask.sum()
            out = out * block_mask.numel() / block_mask.sum()  # 256.128.5.5
            return out

    def _compute_block_mask(self, mask):
        # put = mask[:, None, :, :]#256.1.5.5
        # block_mask  256.1.5.5
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]
        block_mask = 1 - block_mask.squeeze(1)  # 256.5.5
        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)


class LinearScheduler(nn.Module):
    #
    def __init__(self, dropblock, start_value, stop_value, nr_steps):  # 5000
        super(LinearScheduler, self).__init__()
        self.dropblock = dropblock
        self.i = 0
        # np.linspace主要用来创建等差数列
        # 5000
        self.drop_values = np.linspace(start=start_value,
                                       stop=stop_value, num=int(nr_steps))
        # print('k1')

    def forward(self, x):
        return self.dropblock(x)

    def step(self):
        if self.i < len(self.drop_values):
            self.dropblock.drop_prob = self.drop_values[self.i]

        self.i += 1


class FTINet(nn.Module):
    def __init__(self, input_channels, num_nodes, num_classes, patch_size, drop_prob=0.1, block_size=3):
        super(FTINet, self).__init__()
        self.input_channels = input_channels  # 32
        self.num_node = num_nodes  # 17
        self.num_classes = num_classes  # 17
        self.patch_size = patch_size  # 19
        self.dropblock = LinearScheduler(DropBlock2D(drop_prob=drop_prob, block_size=block_size),
                                         start_value=0.,
                                         stop_value=drop_prob,
                                         nr_steps=5e3)

        # bone
        # self.conv1 = nn.Conv2d(self.input_channels, 64, 5)
        self.CIformer1 = CIformer(64, 3)
        self.bnCIformer1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(self.input_channels, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)

        self.CIformer2 = CIformer(64, 3)
        self.bnCIformer2 = nn.BatchNorm2d(64)
        self.CIformer3 = CIformer(64, 3)
        self.bnCIformer3 = nn.BatchNorm2d(64)
        self.model1 = AFLM()
        self.model2 = AFLM()

        self.IFB11 = nn.Conv2d(128, 64, 1)
        self.bn_IFB11 = nn.BatchNorm2d(64)

        self.IFB21 = nn.Conv2d(128, 64, 1)
        self.bn_IFB21 = nn.BatchNorm2d(64)

        self.IFB22 = nn.Conv2d(128, 64, 1)
        self.bn_IFB22 = nn.BatchNorm2d(64)

        self.conv9 = nn.Conv2d(64, 64, 5)
        self.bn9 = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64, 64, 5)
        self.bn10 = nn.BatchNorm2d(64)
        self.model3 = ECM(img_sixe=11)
        self.model4 = ECM(img_sixe=7)

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool1 = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(256, 64)
        self.bn_f1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, self.num_classes)

    def forward(self, x):
        x = x.squeeze(1)

        self.dropblock.step()
        x1 = F.leaky_relu(self.conv2(x))
        x1 = self.bn2(x1)

        x2 = self.CIformer1(x1)
        x2 = self.bnCIformer1(x2)
        x31 = self.CIformer2(x2)
        x31 = self.bnCIformer2(x31)
        x61 = self.CIformer3(x31)
        x61 = self.bnCIformer3(x61)

        result11 = self.model1(x61)

        result21 = self.IFB21(torch.cat((x61, result11), dim=1))
        result21 = self.bn_IFB21(result21)
        result21 = self.model3(result21)

        result12 = self.IFB11(torch.cat((result11, result21), dim=1))
        result12 = self.bn_IFB11(result12)

        result13 = self.model2(result12)
        result13 = F.leaky_relu(self.conv9(result13))
        result13 = self.bn9(result13)

        result22 = self.IFB22(torch.cat((result12, result21), dim=1))
        result22 = self.bn_IFB22(result22)

        result23 = self.model4(result22)

        result23 = F.leaky_relu(self.conv10(result23))
        result23 = self.bn10(result23)
        x_conbine = torch.cat((result13, result23), dim=1)
        x_9 = self.avgpool1(x_conbine)
        x_10 = self.maxpool1(x_conbine)

        x_result = torch.cat((x_10, x_9), dim=-1)
        x_result = x_result.view(-1, x_result.size(1) * x_result.size(2) * x_result.size(3))

        x = F.leaky_relu(self.fc1(x_result))
        x = self.bn_f1(x)

        x = F.leaky_relu(self.fc2(x))

        return x, x



