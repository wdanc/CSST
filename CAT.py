import math
import logging
import collections.abc
from functools import partial

import torch
import torch.nn as nn
from torch import linalg as LA

from timm.models.layers import DropPath
from timm.models.layers import create_conv2d, create_pool2d, to_ntuple, to_2tuple

from models.weights_init import trunc_normal_
from models.helpers import named_apply

_logger = logging.getLogger(__name__)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    This is much like `.vision_transformer.Attention` but uses *localised* self attention by accepting an input with
     an extra "image block" dim
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        x is shape: B (batch_size), T (image blocks), N (seq length per image block), C (embed dim)
        """
        B, T, N, C = x.shape
        # result of next line is (qkv, B, num (H)eads, T, N, (C')hannels per head)
        qkv = self.qkv(x).reshape(B, T, N, 3, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 2, 3, 4, 1).reshape(B, T, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x  # (B, T, N, C)

class TransformerLayer(nn.Module):
    """
    This is much like `.vision_transformer.Block` but:
        - Called TransformerLayer here to allow for "block" as defined in the paper ("non-overlapping image blocks")
        - Uses modified Attention layer that handles the "block" dimension
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.drop_path(self.attn(y))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class CosineCrossAttention(nn.Module):
    def __init__(self, dim, proj_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = proj_dim

        self.to_q = nn.Linear(dim, proj_dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, proj_dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, proj_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(proj_dim, proj_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, kv):
        B, T1, N, _ = query.shape
        _, T2, _, _ = kv.shape
        q = self.to_q(query).reshape(B, T1, N, self.num_heads, self.dim // self.num_heads).permute(0, 3, 1, 2, 4) # B H T N C'
        k = self.to_k(kv).reshape(B, T2, 1, self.num_heads, self.dim // self.num_heads).permute(0, 3, 1, 2, 4) # B H 1 1 C'
        v = self.to_v(kv).reshape(B, T2, 1, self.num_heads, self.dim // self.num_heads).permute(0, 3, 1, 2, 4) # B H 1 1 C'

        norm_q = LA.norm(q, dim=-1, keepdim=True)  # B H T N 1
        norm_k = LA.norm(k, dim=-1, keepdim=True)  # B H 1 1 1
        norm_qk = norm_q @ norm_k  # B H T N 1
        attn = (q @ k.transpose(-2, -1)) / norm_qk # B H T N 1
        attn = self.attn_drop(attn)
        # B H T N C' -> B T N C' H -> B T N C
        x = (attn @ v).permute(0, 2, 3, 4, 1).reshape(B, T1, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CATransformerBlock(nn.Module):
    def __init__(self, dim, proj_dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(proj_dim)
        self.norm2 = norm_layer(proj_dim)
        self.cross_attn = CosineCrossAttention(
            dim, proj_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm3 = norm_layer(proj_dim)
        self.attn = Attention(
            proj_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        mlp_hidden_dim = int(proj_dim * mlp_ratio)
        self.mlp = Mlp(in_features=proj_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, q, kv):
        out = self.norm1(q + self.drop_path(self.cross_attn(q, kv)))
        out = self.norm2(out + self.drop_path(self.attn(out)))
        out = self.norm3(out + self.drop_path(self.mlp(out)))
        return out


class ConvPool(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, pad_type=''):
        super().__init__()
        self.conv = create_conv2d(in_channels, out_channels, kernel_size=3, padding=pad_type, bias=True)
        self.norm = norm_layer(out_channels)
        self.pool = create_pool2d('max', kernel_size=3, stride=2, padding=pad_type)

    def forward(self, x):
        """
        x is expected to have shape (B, C, H, W)
        """
        assert x.shape[-2] % 2 == 0, 'BlockAggregation requires even input spatial dims'
        assert x.shape[-1] % 2 == 0, 'BlockAggregation requires even input spatial dims'
        x = self.conv(x)
        # Layer norm done over channel dim only
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.pool(x)
        return x  # (B, C, H//2, W//2)


class LocalNestLevel(nn.Module):
    """ Single hierarchical level of a Nested Transformer
    """

    def __init__(
            self, block_size, num_heads, depth, embed_dim, prev_embed_dim=None,
            mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rates=[],
            norm_layer=None, act_layer=None, pad_type=''):
        super().__init__()
        self.block_size = block_size

        if prev_embed_dim is not None:
            self.pool = ConvPool(prev_embed_dim, embed_dim, norm_layer=norm_layer, pad_type=pad_type)
        else:
            self.pool = nn.Identity()

        # Transformer encoder
        if len(drop_path_rates):
            assert len(drop_path_rates) == depth, 'Must provide as many drop path rates as there are transformer layers'
        self.transformer_encoder = nn.Sequential(*[
            TransformerLayer(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rates[i],
                norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

    def forward(self, x):
        """
        expects x as (B, C, H, W)
        """
        x = self.pool(x)
        x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
        x = blockify(x, self.block_size)  # (B, T, N, C')
        x = self.transformer_encoder(x)  # (B, T, N, C')

        x = deblockify(x, self.block_size)  # (B, H', W', C')
        # Channel-first for block aggregation, and generally to replicate convnet feature map at each stage
        return x.permute(0, 3, 1, 2)


class CATransformerLevel(nn.Module):
    def __init__(
            self, block_size, num_heads, depth, embed_dim,
            mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
            norm_layer=None, act_layer=None):
        super().__init__()
        self.block_size = block_size
        self.depth = depth
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        for i in range(depth):
            self.add_module('d_extract_'+str(i), nn.Conv2d(embed_dim, 2, kernel_size=3, padding=1))
            self.add_module('d_transformer_'+str(i), CATransformerBlock(
                dim=embed_dim, proj_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate,
                norm_layer=norm_layer, act_layer=act_layer))


    def forward(self, x):
        p = []
        for i in range(self.depth):
            d_extract = getattr(self, 'd_extract_'+str(i))
            d_transformer = getattr(self, 'd_transformer_'+str(i))
            d = d_extract(x)
            p.append(d)
            mask = d.softmax(dim=1)
            kv = x * mask[:, 1:]
            kv = self.global_pool(kv)
            kv = kv.permute(0, 2, 3, 1)  # B 1 1 C

            x = x.permute(0, 2, 3, 1)  # (B, H', W', C), switch to channels last for transformer
            x = blockify(x, self.block_size)  # (B, T, N, C')

            x = d_transformer(x, kv)
            x = deblockify(x, self.block_size)
            x = x.permute(0, 3, 1, 2)

        return x, p


def blockify(x, block_size: int):
    """image to blocks
    Args:
        x (Tensor): with shape (B, H, W, C)
        block_size (int): edge length of a single square block in units of H, W
    """
    B, H, W, C = x.shape
    assert H % block_size == 0, '`block_size` must divide input height evenly'
    assert W % block_size == 0, '`block_size` must divide input width evenly'
    grid_height = H // block_size
    grid_width = W // block_size
    x = x.reshape(B, grid_height, block_size, grid_width, block_size, C)
    x = x.transpose(2, 3).reshape(B, grid_height * grid_width, -1, C)
    # print(H, W, C, block_size, x.shape)  block size=8
    return x  # (B, T, N, C)


def deblockify(x, block_size: int):
    """blocks to image
    Args:
        x (Tensor): with shape (B, T, N, C) where T is number of blocks and N is sequence size per block
        block_size (int): edge length of a single square block in units of desired H, W
    """
    B, T, _, C = x.shape
    grid_size = int(math.sqrt(T))
    height = width = grid_size * block_size
    x = x.reshape(B, grid_size, grid_size, block_size, block_size, C)
    x = x.transpose(2, 3).reshape(B, height, width, C)
    return x  # (B, H, W, C)


class Upsample(nn.Module):
    def __init__(self, ups_factor, in_chans, up_chans):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_chans, up_chans, kernel_size=1, stride=1, padding=0)
        self.pixel_shuffle = nn.PixelShuffle(ups_factor)
        self.norm = nn.BatchNorm2d(up_chans//ups_factor**2)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.act(x)
        x = self.norm(x)
        return x


class CAT(nn.Module):
    def __init__(self, img_size=256, in_chans=3, patch_size=4, num_levels=3, block_size=8, embed_dims=(96, 192, 384),
                 num_heads=(3, 6, 12), depths=(4, 4, 6), dt_depth=(2,2,2), num_classes=2, mlp_ratio=4, qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0, norm_layer=None, act_layer=None,
                 pad_type='', weight_init=''):
        super(CAT, self).__init__()

        self.num_levels = num_levels
        for param_name in ['embed_dims', 'num_heads', 'depths']:
            param_value = locals()[param_name]
            if isinstance(param_value, collections.abc.Sequence):
                assert len(param_value) == self.num_levels, f'Require `len({param_name}) == num_levels`'

        embed_dims = to_ntuple(self.num_levels)(embed_dims)
        num_heads = to_ntuple(self.num_levels)(num_heads)
        depths = to_ntuple(self.num_levels)(depths)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.drop_rate = drop_rate
        if isinstance(img_size, collections.abc.Sequence):
            assert img_size[0] == img_size[1], 'Model only handles square inputs'
            img_size = img_size[0]
        assert img_size % patch_size == 0, '`patch_size` must divide `img_size` evenly'
        self.patch_size = patch_size

        self.block_size = block_size
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dims[0], flatten=False)
        self.num_patches = self.patch_embed.num_patches

        # Build up each hierarchical level
        levels = []
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        prev_dim = None
        for i in range(self.num_levels):
            dim = embed_dims[i]
            levels.append(LocalNestLevel(
                self.block_size, num_heads[i], depths[i], dim, prev_dim,
                mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, dp_rates[i], norm_layer, act_layer, pad_type=pad_type))
            prev_dim = dim
        self.levels = nn.Sequential(*levels)

        diff_levels = []
        for i in range(self.num_levels):
            dim = embed_dims[i]
            diff_levels.append(CATransformerLevel(
                self.block_size, num_heads[i], dt_depth[i], dim,
                mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, norm_layer, act_layer))
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

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        named_apply(partial(_init_nest_weights, head_bias=head_bias), self)

    def forward(self, x1, x2):
        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)
        diffs = []
        p = []
        for i in range(self.num_levels):
            x1 = self.levels[i](x1)
            x2 = self.levels[i](x2)
            diff = torch.cat((x2, x1), dim=1)
            fuse = getattr(self, 'diff_fuse'+str(i))
            diff = fuse(diff) + (x2-x1)
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


def _init_nest_weights(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ NesT weight initialization
    Can replicate Jax implementation. Otherwise follows vision_transformer.py
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            trunc_normal_(module.weight, std=.02, a=-2, b=2)
            nn.init.constant_(module.bias, head_bias)
        else:
            trunc_normal_(module.weight, std=.02, a=-2, b=2)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        trunc_normal_(module.weight, std=.02, a=-2, b=2)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
