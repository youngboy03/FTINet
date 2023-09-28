# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model


import numpy as np

import torch

from torch.nn import Sequential as Seq, Linear as Lin, Conv2d


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'ECM': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'ECM': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

def get_2d_relative_pos_embed(embed_dim, grid_size):

    #embed_dim=32
    #grid_size=19
    pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size)#361.32
    relative_pos = 2 * np.matmul(pos_embed, pos_embed.transpose()) / pos_embed.shape[1]#361.361
    return relative_pos

# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):

    grid_h = np.arange(grid_size, dtype=np.float32)#0到18个数字
    grid_w = np.arange(grid_size, dtype=np.float32)#0到18个数字
    grid = np.meshgrid(grid_w, grid_h)# 生成网格grid点坐标矩阵 # here w goes first 这里 w 先行
    grid = np.stack(grid, axis=0)#2.19.19这个函数的作用就是堆叠作用，就是将两个分离的数据堆叠到一个数据里

    grid = grid.reshape([2, 1, grid_size, grid_size])#2.1.19.19
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)#361.32
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)#np.concatenate()是用来对数列或矩阵进行合并的
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])#361.16  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])#361.16  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)#361.32 # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)#0到7 8个数
    omega /= embed_dim / 2.#数组里面的数字都除以8
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega) #361.8 # (M, D/2), outer product 外积

    emb_sin = np.sin(out) # (M, D/2)#361.8
    emb_cos = np.cos(out) # (M, D/2)#361.8

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)#361.16
    return emb


def pairwise_distance(x):#2.361.32

    with torch.no_grad():
        x_inner = -2*torch.matmul(x, x.transpose(2, 1))#2.361.361
        #对两个张量进行逐元素乘法
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)#2.361.1
        return x_square + x_inner + x_square.transpose(2, 1)

def dense_knn_matrix(x, k=16, relative_pos=None):

    #x=2.32.361.1    relative_pos=1.361.361
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)#2.361.32
        batch_size, n_points, n_dims = x.shape


        #dist = relative_pos.repeat(batch_size,1,1)绝对距离
        dist = pairwise_distance(x.detach())#2.361.361
        if relative_pos is not None:
            dist += relative_pos#256.121.121
        _, nn_idx = torch.topk(-dist, k=k)#256.121.20 # b, n, k  取一个tensor的topk元素  返回沿给定维度的给定 input 张量的 k 个最大元素
            #返回 (values, indices) namedtuple ，其中 indices 是原始 input 张量中元素的索引。
        ######).repea函数可以对张量进行重复扩充
    peat=torch.arange(0, n_points, device=x.device)
    center_idx = peat.repeat(batch_size, k, 1).transpose(2, 1)#256.121.30
    p=torch.stack((nn_idx, center_idx), dim=0)
    return p

class DenseDilated(nn.Module):

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation   #1
        self.stochastic = stochastic  #false
        self.epsilon = epsilon   #0.2
        self.k = k   #9

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, :, randnum]
            else:
                edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index

class DenseDilatedKnnGraph(nn.Module):

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation   #1
        self.stochastic = stochastic  #Flase
        self.epsilon = epsilon  #0.2
        self.k = k   #9
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)

    def forward(self, x, y=None, relative_pos=None):

        x = F.normalize(x, p=2.0, dim=1)#将某一个维度除以那个维度对应的范数(默认是2范数)。2.32.361.1表示得是361是H和W相乘
        edge_index = dense_knn_matrix(x, self.k * self.dilation, relative_pos)#2.2.361.9
        return self._dilated(edge_index)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer

    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def norm_layer(norm, nc):
    # normalization layer 2d
    norm = norm.lower()
    if norm == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return layer

class BasicConv(Seq):
    def __init__(self, channels, act='relu', norm=None, bias=True, drop=0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=bias, groups=4))
            if norm is not None and norm.lower() != 'none':
                m.append(norm_layer(norm, channels[-1]))
            if act is not None and act.lower() != 'none':
                m.append(act_layer(act))
            if drop > 0:
                m.append(nn.Dropout2d(drop))

        super(BasicConv, self).__init__(*m)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def batched_index_select(x, idx):

    #x=2.32.361.1   #2.361.9
    x1=idx
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base=torch.arange(0, batch_size, device=idx.device)
    idx_base = idx_base.view(-1, 1, 1) * num_vertices_reduced#2.1.1
    #b=idx_base   0    225
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)#6498
    x = x.transpose(2, 1)  # 2.361.32.1
    x = x.contiguous().view(batch_size * num_vertices_reduced, -1)#450.64
    feature = x[idx, :]  # 2.32.361.9
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature

class EdgeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(EdgeConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels], act, norm, bias)

    def forward(self, x, edge_index, y=None):
        x1=edge_index[0]
        x2=edge_index[1]

        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        mean_value = torch.mean(self.nn(torch.cat([x_i, x_j - x_i], dim=1)), -1, keepdim=True)
        return mean_value


class GraphConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, conv='edge', act='relu', norm=None, bias=True):
        super(GraphConv2d, self).__init__()#选择的是mr
        if conv == 'edge':
            self.gconv = EdgeConv2d(in_channels, out_channels, act, norm, bias)
        else:
            raise NotImplementedError('conv:{} is not supported'.format(conv))

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)

class DyGraphConv2d(GraphConv2d):

    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, stochastic=False, epsilon=0.0, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels, conv, act, norm, bias)
        self.k = kernel_size    #9
        self.d = dilation      #1
        self.r = r        #1
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)#密集扩张 Knn 图

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape#2.32.19.19
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()#2.32.361.1
        edge_index = self.dilated_knn_graph(x, y, relative_pos)#2.2.361.9
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)#2.64.361.1
        return x.reshape(B, -1, H, W).contiguous()


class Grapher(nn.Module):

    def __init__(self, in_channels, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True,  stochastic=False, epsilon=0.0, r=1, n=196, drop_path=0.0, relative_pos=False):
        super(Grapher, self).__init__()
        self.channels = in_channels #32
        self.n = n   #HW 19*19
        self.r = r   #1
        self.layer1 = nn.GELU()

        #一层图卷积
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, conv,
                              act, norm, bias, stochastic, epsilon, r)
        #在一层卷积层
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )
        #输入是啥，直接给输出，不做任何的改变 nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.relative_pos = None  #使用相对位置
        if relative_pos:
            print('using relative_pos')
            #方法把数组转换成张量，且二者共享内存，对张量进行修改比如重新赋值，那么原始数组也会相应发生改变。
            #下面这一行是求相对位置的1.1.361.361
            relative_pos_tensor = torch.from_numpy(np.float32(get_2d_relative_pos_embed(in_channels,
                int(n**0.5)))).unsqueeze(0).unsqueeze(1)
            #利用插值方法，对输入的张量数组进行上\下采样操作，换句话说就是科学合理地改变数组的尺寸大小，尽量保持数据完整
            relative_pos_tensor = F.interpolate(
                    relative_pos_tensor, size=(n, n//(r*r)), mode='bicubic', align_corners=False)
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            #liu=2   debug之后进来了
            return relative_pos
        else:
            N = H * W
            N_reduced = N // (self.r * self.r)
            return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    def forward(self, x):
        _tmp = x  #2.32.19.19
        B, C, H, W = x.shape  #2.32.19.19
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)#1.361.361
        #x=2.32.19.19    relative_pos=1.361.361
        x = self.graph_conv(x, relative_pos)#2.64.19.19
        x=self.layer1(x)
        x = self.fc2(x)#2.32.19.19
        x = self.drop_path(x) + _tmp#2.32.19.19
        return x




class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        print(opt)
        k = opt.k  #9
        act = opt.act  #gelu
        norm = opt.norm  #batch
        bias = opt.bias   #true
        epsilon = opt.epsilon   #0.2gcn的随机ε
        stochastic = opt.use_stochastic#flase
        conv = opt.conv#mr

        blocks = opt.blocks#[2.2.18.2]
        self.n_blocks = sum(blocks)#21
        channels = opt.channels#[32.128.320.512]
        self.img_size = opt.img_size#19

        HW = self.img_size * self.img_size  #361


        self.backbone_1 = nn.ModuleList([])
        self.number = [2]
        for j in range(self.number[0]):
            self.backbone_1 += [
                Seq(Grapher(channels[0], k, 1, conv, act, norm,
                            bias, stochastic, epsilon, 1, n=HW, drop_path=0,
                            relative_pos=True))]
        self.backbone_1 = Seq(*self.backbone_1)

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)#使用正态分布对输入张量进行赋值
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        for i in range(len(self.backbone_1)):
            x = self.backbone_1[i](inputs)  #(2,128,64,64)(2,128,64,64)(2,128,32,32)(2,128,32,32)(2,128,16,16)i=24(2,1024,8,8)

        return x





def ECM(pretrained=False, **kwargs):
    class OptInit:
        def __init__(self, num_classes=2, drop_path_rate=0.0,img_sixe=7, **kwargs):
            self.k = 12 # neighbor num (default:9)
            self.conv = 'edge'  # graph conv layer {edge, mr,sage,gin}
            self.act = 'gelu'  # activation layer {relu, prelu, leakyrelu, gelu, hswish}激活层
            self.norm = 'batch'  # batch or instance normalization {batch, instance}批量或实例规范化
            self.bias = True  # bias of conv layer True or False
            self.dropout = 0.0  # dropout rate
            self.use_dilation = True  # use dilated knn or not
            self.epsilon = 0.2  # stochastic epsilon for gcn  gcn的随机ε
            self.use_stochastic = False  # stochastic for gcn, True or False
            self.drop_path = drop_path_rate
            self.blocks = [2, 2]  # number of basic blocks in the backbone主干中的基本块数
            self.channels = [64, 128]  # number of channels of deep features[48, 96, 240, 384][128, 256, 512, 1024]深度特征的通道数
            self.n_classes = num_classes  # Dimension of out_channelsout_channels 的维度
            self.emb_dims = 1024  # Dimension of embeddings 嵌入的维度
            self.img_size = img_sixe


    opt = OptInit(**kwargs)
    model = DeepGCN(opt)
    model.default_cfg = default_cfgs['ECM']
    return model


if __name__ == "__main__":
    model = ECM()
    a = torch.randn((2, 64, 15, 15))
    result = model(a)
    print(result.shape)