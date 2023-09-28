
import torch
from torch import einsum, nn
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import math
import numpy as np
from einops import rearrange

from timm.models.layers import DropPath, trunc_normal_




from mmcv.runner import (BaseModule, ModuleList, load_checkpoint)


class DWConv2d_BN(nn.Module):


    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
    ):
        super().__init__()

        # dw
        self.dwconv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size,
            stride,
            (kernel_size - 1) // 2,
            groups=out_ch,
            bias=False,
        )
        # pw-linear
        self.pwconv = nn.Conv2d(out_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(bn_weight_init)
                m.bias.data.zero_()

    def forward(self, x):#1.32.56.56
        x = self.dwconv(x)#1.32.56.56
        x = self.pwconv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DWCPatchEmbed(nn.Module):

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 act_layer=nn.Hardswish):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
        )

    def forward(self, x):#1.32.56.56

        x = self.patch_conv(x)#1.32.56.56

        return x


class Patch_Embed_stage(nn.Module):

    def __init__(self, embed_dim, num_path=4, isPool=False, stage=0):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList([
            DWCPatchEmbed(
                in_chans=embed_dim,
                embed_dim=embed_dim,
                patch_size=3,
                stride=1
            )
        ])

    def forward(self, x):
        #att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)

            #att_inputs.append(x)

        return x


class LowPassModule(nn.Module):
    def __init__(self, in_channel, sizes=(3, 5)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])
        self.relu = nn.ReLU()
        ch =  in_channel // 2
        self.channel_splits = [ch, ch]

    def _make_stage(self, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return nn.Sequential(prior)

    def forward(self, feats):#1.64.112.112
        h, w = feats.size(2), feats.size(3)
        feats = torch.split(feats, self.channel_splits, dim=1)#4ge 1.16.112.112
        priors = [F.upsample(input=self.stages[i](feats[i]), size=(h, w), mode='bilinear') for i in range(2)]
        bottle = torch.cat(priors, 1)

        return self.relu(bottle)

class FilterModule(nn.Module):
    def __init__(self, Ch, h, window):#8,8,[3:2,5:3.7:3]

        super().__init__()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch,
                cur_head_split * Ch,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split * Ch,
            )
            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch for x in self.head_splits]
        self.LP = LowPassModule(Ch * h)

    def forward(self, q, v, size):
        B, h, N, Ch = q.shape
        H, W = size

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        v_img = rearrange(v, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)#1.64.112.112
        LP = self.LP(v_img)
        #1.64.112.112
        # Split according to channels.
        # 16.24.24
        v_img_list = torch.split(v_img, self.channel_splits, dim=1)
        HP_list = [
            conv(x) for conv, x in zip(self.conv_list, v_img_list)
        ]
        HP = torch.cat(HP_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        HP = rearrange(HP, "B (h Ch) H W -> B h (H W) Ch", h=h)
        LP = rearrange(LP, "B (h Ch) H W -> B h (H W) Ch", h=h)

        dynamic_filters = q * HP + LP
        return dynamic_filters

class AFLM(BaseModule):
    def __init__(
        self,
        num_stages=4,
        num_path=[4, 4, 4, 4],
        embed_dims=[64, 128, 256, 512],
        num_heads=[8, 8, 8, 8],
        num_classes=16,
        pretrained=None,
        crpe_window={
            3: 4,
            5: 4
        }
    ):
        super().__init__()
        if isinstance(pretrained, str):
            self.init_cfg = pretrained
        self.num_classes = num_classes
        self.num_stages = num_stages



        self.patch_embed_stages = nn.ModuleList([
            Patch_Embed_stage(
                embed_dims[idx],
                num_path=num_path[idx],
                isPool=True if idx == 1 else False,
                stage=idx,
            ) for idx in range(self.num_stages)
        ])

        self.qkv = nn.Linear(embed_dims[0], embed_dims[0] * 3, bias=False)

        self.num_heads = num_heads[0]
        head_dim = embed_dims[0] // num_heads[0]
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(embed_dims[0], embed_dims[0] * 3, bias=False)

        self.proj = nn.Linear(embed_dims[0], embed_dims[0])


        self.crpe = FilterModule(Ch=64 // 8,
                                 h=num_heads[0],
                                 window=crpe_window)




    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()



    def forward(self, x):
        out = []

        #att_inputs = self.patch_embed_stages[idx](x)

        for idx in range(self.num_stages):
            x= self.patch_embed_stages[idx](x)
            B,C,H,W=x.shape
            x=x.permute(0, 1, 2, 3).reshape(B, H*W, -1)
            B, N, C = x.shape  # 1.12544.64

            # Generate Q, K, V.#3.1.8.12544.8
            qkv = (self.qkv(x).reshape(B, N, 3, self.num_heads,
                                       C // self.num_heads).permute(2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Factorized attention.
            k_softmax = k.softmax(dim=2)
            k_softmax_T_dot_v = einsum("b h n k, b h n v -> b h k v", k_softmax, v)
            factor_att = einsum("b h n k, b h k v -> b h n v", q,
                                k_softmax_T_dot_v)

            # Convolutional relative position encoding.
            # 1.8.12544.8
            crpe = self.crpe(q, v, size=(H,W))

            # Merge and reshape.
            x = self.scale * factor_att + crpe
            x = x.transpose(1, 2).reshape(B, N, C)

            # Output projection.
            x = self.proj(x)
            x = x.permute(0, 2, 1).reshape(B, C,H , W)


        return x






class AFLM(AFLM):
    def __init__(self, **kwargs):
        super(AFLM, self).__init__(
            num_stages=1,
            num_path=[1],
            embed_dims=[64],
            num_heads=[8], **kwargs)

if __name__ == "__main__":
    model = AFLM()
    a = torch.randn((16, 64, 19, 19))
    result = model(a)
    print(result)