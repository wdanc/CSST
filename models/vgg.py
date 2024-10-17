import torch
import torch.nn as nn
import math
import collections
from timm.models.layers import to_ntuple
from models.CSST import CSSTransformerLevel, Upsample



def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer


class VGG16(nn.Module):
    def __init__(self, ):
        super(VGG16, self).__init__()

        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        # self.layer6 = vgg_fc_layer(7 * 7 * 512, 4096)
        # self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        # self.layer8 = tnn.Linear(4096, n_classes)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # vgg16_features = self.layer5(out)

        return [x2, x3, x4]


class CSST_vgg(nn.Module):
    def __init__(self, vgg_dims=(128,256,512), num_levels=3, block_size=8, embed_dims=(96, 192, 384),
                 num_heads=(3, 6, 12), dt_depths=(2, 2, 2), num_classes=2, mlp_ratio=4, qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., act_layer=None,
                 ):
        super(CSST_vgg, self).__init__()

        self.vgg = VGG16()

        self.num_levels = num_levels
        for param_name in ['embed_dims', 'num_heads', 'dt_depths']:
            param_value = locals()[param_name]
            if isinstance(param_value, collections.abc.Sequence):
                assert len(param_value) == self.num_levels, f'Require `len({param_name}) == num_levels`'

        embed_dims = to_ntuple(self.num_levels)(embed_dims)
        num_heads = to_ntuple(self.num_levels)(num_heads)
        # self.num_classes = num_classes
        self.num_features = embed_dims[-1]
        dt_norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.drop_rate = drop_rate

        self.block_size = block_size

        diff_levels = []
        for i in range(self.num_levels):
            self.add_module('chans_proj'+str(i), nn.Conv2d(vgg_dims[i], embed_dims[i], kernel_size=1))
            dim = embed_dims[i]
            diff_levels.append(CSSTransformerLevel(
                self.block_size, num_heads[i], dt_depths[i], dim,
                mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dt_norm_layer, act_layer))
        self.diff_levels = nn.Sequential(*diff_levels)

        for i in range(self.num_levels):
            self.add_module('diff_fuse' + str(i), nn.Conv2d(embed_dims[i] * 2, embed_dims[i],
                                                            kernel_size=3, stride=1, padding=1))
            for j in range(self.num_levels - 1, i, -1):
                self.add_module('upsample_' + str(j) + str(i),
                                Upsample(ups_factor=2 ** (j - i), in_chans=embed_dims[j],
                                         up_chans=embed_dims[i] * (2 ** (j - i)) ** 2))

        self.upsample_00 = Upsample(ups_factor=4, in_chans=embed_dims[0], up_chans=embed_dims[0] * 16)

        self.head = nn.Sequential(
            nn.Conv2d(embed_dims[0], 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x1, x2):
        x1s = self.vgg(x1)
        x2s = self.vgg(x2)
        diffs = []
        p = []
        for i in range(self.num_levels):
            chans_proj = getattr(self, 'chans_proj'+str(i))
            x1 = x1s[i]
            x2 = x2s[i]
            d_x1 = chans_proj(x1)
            d_x2 = chans_proj(x2)
            diff = torch.cat((d_x2, d_x1), dim=1)
            fuse = getattr(self, 'diff_fuse'+str(i))
            diff = fuse(diff) + torch.abs(d_x2-d_x1)
            diff, media_p = self.diff_levels[i](diff)
            diffs.append(diff)
            p.extend(media_p)

        for i in range(self.num_levels - 1, 0, -1):
            for j in range(i):
                upsample = getattr(self, 'upsample_' + str(i) + str(j))
                diffs[j] = diffs[j] + upsample(diffs[i])

        diffs[0] = self.upsample_00(diffs[0])
        diff = self.head(diffs[0])

        return diff, p