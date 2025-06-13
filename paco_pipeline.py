from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from pointnet2_ops import pointnet2_utils
from pytorch3d.structures import Pointclouds
from pytorch3d.loss import chamfer_distance
from scipy.optimize import linear_sum_assignment

from . import MODELS
from .transformer_utils import (
    LayerScale, MLP, Attention, DeformableLocalAttention,
    DeformableLocalCrossAttention, DynamicGraphAttention,
    ImprovedDeformableLocalGraphAttention, CrossAttention,
    knn_point, index_points
)

from .diffusion_components import VectorQuantizer, NoiseScheduler, DiffusionUNet, CrossLevelAttention
from typing import List, Dict, Tuple
class SelfAttnBlockAPI(nn.Module):
    r"""
    Self Attention Block API

    This block supports different configurations:
      1. Norm Encoder Block (block_style = 'attn')
      2. Concatenation Fused Encoder Block (block_style = 'attn-deform', combine_style = 'concat')
      3. Three-layer Fused Encoder Block (block_style = 'attn-deform', combine_style = 'onebyone')
    """

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            block_style='attn-deform',
            combine_style='concat',
            k=10,
            n_group=2
    ):
        super().__init__()
        self.combine_style = combine_style
        # Ensure the combine style is valid
        assert combine_style in ['concat', 'onebyone'], (
            f'got unexpect combine_style {combine_style} for local and global attn'
        )
        self.norm1 = norm_layer(dim)
        # Apply LayerScale if init_values is provided, else identity
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Process block style tokens (e.g. 'attn' or 'attn-deform')
        block_tokens = block_style.split('-')
        assert 0 < len(block_tokens) <= 2, f'invalid block_style {block_style}'
        self.block_length = len(block_tokens)
        self.attn = None
        self.local_attn = None
        for block_token in block_tokens:
            assert block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], (
                f'got unexpect block_token {block_token} for Block component'
            )
            if block_token == 'attn':
                self.attn = Attention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop
                )
            elif block_token == 'rw_deform':
                self.local_attn = DeformableLocalAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group
                )
            elif block_token == 'deform':
                self.local_attn = DeformableLocalCrossAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group
                )
            elif block_token == 'graph':
                self.local_attn = DynamicGraphAttention(dim, k=k)
            elif block_token == 'deform_graph':
                self.local_attn = ImprovedDeformableLocalGraphAttention(dim, k=k)
                
        # If both global and local attentions are used, set up merging
        if self.attn is not None and self.local_attn is not None:
            if combine_style == 'concat':
                self.merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, pos, idx=None):
        """
        Forward pass for the SelfAttnBlockAPI

        Args:
            x: Input feature tensor
            pos: Positional encoding tensor
            idx: (Optional) Index tensor for local attention computations

        Returns:
            Output tensor after self-attention and MLP operations
        """
        feature_list = []
        if self.block_length == 2:
            if self.combine_style == 'concat':
                norm_x = self.norm1(x)
                if self.attn is not None:
                    global_attn_feat = self.attn(norm_x)
                    feature_list.append(global_attn_feat)
                if self.local_attn is not None:
                    local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                    feature_list.append(local_attn_feat)
                # Combine features via concatenation and linear mapping
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.merge_map(f)
                    x = x + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path3(self.ls3(self.local_attn(self.norm3(x), pos, idx=idx)))
        elif self.block_length == 1:
            norm_x = self.norm1(x)
            if self.attn is not None:
                global_attn_feat = self.attn(norm_x)
                feature_list.append(global_attn_feat)
            if self.local_attn is not None:
                local_attn_feat = self.local_attn(norm_x, pos, idx=idx)
                feature_list.append(local_attn_feat)
            # If only one attention branch is used, add it directly
            if len(feature_list) == 1:
                f = feature_list[0]
                x = x + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()
        # Apply MLP block after attention
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CrossAttnBlockAPI(nn.Module):
    r"""
    Cross Attention Block API

    This block supports multiple configurations for self-attention and cross-attention.
    Examples of supported designs include:
      1. Norm Decoder Block (self_attn_block_style = 'attn', cross_attn_block_style = 'attn')
      2. Concatenation Fused Decoder Block (self_attn_block_style = 'attn-deform', combine_style = 'concat')
      3. Three-layer Fused Decoder Block (self_attn_block_style = 'attn-deform', combine_style = 'onebyone')
      4. Custom designs mixing plain and deformable attention or graph convolution
    """

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            init_values=None, drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            self_attn_block_style='attn-deform', self_attn_combine_style='concat',
            cross_attn_block_style='attn-deform', cross_attn_combine_style='concat',
            k=10, n_group=2
    ):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Self-attention branch initialization
        self.norm1 = norm_layer(dim)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.self_attn_combine_style = self_attn_combine_style
        assert self_attn_combine_style in ['concat', 'onebyone'], (
            f'got unexpect self_attn_combine_style {self_attn_combine_style} for local and global attn'
        )

        self_attn_block_tokens = self_attn_block_style.split('-')
        assert 0 < len(self_attn_block_tokens) <= 2, f'invalid self_attn_block_style {self_attn_block_style}'
        self.self_attn_block_length = len(self_attn_block_tokens)
        self.self_attn = None
        self.local_self_attn = None
        for self_attn_block_token in self_attn_block_tokens:
            assert self_attn_block_token in ['attn', 'rw_deform', 'deform', 'graph', 'deform_graph'], (
                f'got unexpect self_attn_block_token {self_attn_block_token} for Block component'
            )
            if self_attn_block_token == 'attn':
                self.self_attn = Attention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop
                )
            elif self_attn_block_token == 'rw_deform':
                self.local_self_attn = DeformableLocalAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group
                )
            elif self_attn_block_token == 'deform':
                self.local_self_attn = DeformableLocalCrossAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group
                )
            elif self_attn_block_token == 'graph':
                self.local_self_attn = DynamicGraphAttention(dim, k=k)
            elif self_attn_block_token == 'deform_graph':
                self.local_self_attn = ImprovedDeformableLocalGraphAttention(dim, k=k)
        if self.self_attn is not None and self.local_self_attn is not None:
            if self_attn_combine_style == 'concat':
                self.self_attn_merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm3 = norm_layer(dim)
                self.ls3 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Cross-attention branch initialization
        self.norm_q = norm_layer(dim)
        self.norm_v = norm_layer(dim)
        self.ls4 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cross_attn_combine_style = cross_attn_combine_style
        assert cross_attn_combine_style in ['concat', 'onebyone'], (
            f'got unexpect cross_attn_combine_style {cross_attn_combine_style} for local and global attn'
        )

        cross_attn_block_tokens = cross_attn_block_style.split('-')
        assert 0 < len(cross_attn_block_tokens) <= 2, f'invalid cross_attn_block_style {cross_attn_block_style}'
        self.cross_attn_block_length = len(cross_attn_block_tokens)
        self.cross_attn = None
        self.local_cross_attn = None
        for cross_attn_block_token in cross_attn_block_tokens:
            assert cross_attn_block_token in ['attn', 'deform', 'graph', 'deform_graph'], (
                f'got unexpect cross_attn_block_token {cross_attn_block_token} for Block component'
            )
            if cross_attn_block_token == 'attn':
                self.cross_attn = CrossAttention(
                    dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop
                )
            elif cross_attn_block_token == 'deform':
                self.local_cross_attn = DeformableLocalCrossAttention(
                    dim, num_heads=num_heads, qkv_bias=qkv_bias,
                    attn_drop=attn_drop, proj_drop=drop, k=k, n_group=n_group
                )
            elif cross_attn_block_token == 'graph':
                self.local_cross_attn = DynamicGraphAttention(dim, k=k)
            elif cross_attn_block_token == 'deform_graph':
                self.local_cross_attn = ImprovedDeformableLocalGraphAttention(dim, k=k)
        if self.cross_attn is not None and self.local_cross_attn is not None:
            if cross_attn_combine_style == 'concat':
                self.cross_attn_merge_map = nn.Linear(dim * 2, dim)
            else:
                self.norm_q_2 = norm_layer(dim)
                self.norm_v_2 = norm_layer(dim)
                self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path5 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, q, v, q_pos, v_pos, self_attn_idx=None, cross_attn_idx=None, denoise_length=None):
        """
        Forward pass for the CrossAttnBlockAPI

        Args:
            q: Query tensor
            v: Value tensor
            q_pos: Positional encoding for query
            v_pos: Positional encoding for value
            self_attn_idx: (Optional) Index tensor for self-attention neighborhood
            cross_attn_idx: (Optional) Index tensor for cross-attention neighborhood
            denoise_length: (Optional) Length for denoising tokens

        Returns:
            Updated query tensor after cross-attention and MLP operations
        """
        # Determine mask if denoising is applied
        if denoise_length is None:
            mask = None
        else:
            query_len = q.size(1)
            mask = torch.zeros(query_len, query_len).to(q.device)
            mask[:-denoise_length, -denoise_length:] = 1.

        # Self-attention branch
        feature_list = []
        if self.self_attn_block_length == 2:
            if self.self_attn_combine_style == 'concat':
                norm_q = self.norm1(q)
                if self.self_attn is not None:
                    global_attn_feat = self.self_attn(norm_q, mask=mask)
                    feature_list.append(global_attn_feat)
                if self.local_self_attn is not None:
                    local_attn_feat = self.local_self_attn(
                        norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length
                    )
                    feature_list.append(local_attn_feat)
                # Combine self-attention features
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.self_attn_merge_map(f)
                    q = q + self.drop_path1(self.ls1(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                q = q + self.drop_path1(self.ls1(self.self_attn(self.norm1(q), mask=mask)))
                q = q + self.drop_path3(self.ls3(
                    self.local_self_attn(self.norm3(q), q_pos, idx=self_attn_idx, denoise_length=denoise_length)
                ))
        elif self.self_attn_block_length == 1:
            norm_q = self.norm1(q)
            if self.self_attn is not None:
                global_attn_feat = self.self_attn(norm_q, mask=mask)
                feature_list.append(global_attn_feat)
            if self.local_self_attn is not None:
                local_attn_feat = self.local_self_attn(
                    norm_q, q_pos, idx=self_attn_idx, denoise_length=denoise_length
                )
                feature_list.append(local_attn_feat)
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path1(self.ls1(f))
            else:
                raise RuntimeError()

        # Cross-attention branch
        feature_list = []
        if self.cross_attn_block_length == 2:
            if self.cross_attn_combine_style == 'concat':
                norm_q = self.norm_q(q)
                norm_v = self.norm_v(v)
                if self.cross_attn is not None:
                    global_attn_feat = self.cross_attn(norm_q, norm_v)
                    feature_list.append(global_attn_feat)
                if self.local_cross_attn is not None:
                    local_attn_feat = self.local_cross_attn(
                        q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx
                    )
                    feature_list.append(local_attn_feat)
                if len(feature_list) == 2:
                    f = torch.cat(feature_list, dim=-1)
                    f = self.cross_attn_merge_map(f)
                    q = q + self.drop_path4(self.ls4(f))
                else:
                    raise RuntimeError()
            else:  # onebyone
                q = q + self.drop_path4(self.ls4(self.cross_attn(self.norm_q(q), self.norm_v(v))))
                q = q + self.drop_path5(self.ls5(
                    self.local_cross_attn(
                        q=self.norm_q_2(q), v=self.norm_v_2(v), q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx
                    )
                ))
        elif self.cross_attn_block_length == 1:
            norm_q = self.norm_q(q)
            norm_v = self.norm_v(v)
            if self.cross_attn is not None:
                global_attn_feat = self.cross_attn(norm_q, norm_v)
                feature_list.append(global_attn_feat)
            if self.local_cross_attn is not None:
                local_attn_feat = self.local_cross_attn(
                    q=norm_q, v=norm_v, q_pos=q_pos, v_pos=v_pos, idx=cross_attn_idx
                )
                feature_list.append(local_attn_feat)
            if len(feature_list) == 1:
                f = feature_list[0]
                q = q + self.drop_path4(self.ls4(f))
            else:
                raise RuntimeError()

        q = q + self.drop_path2(self.ls2(self.mlp(self.norm2(q))))
        return q


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder without hierarchical structure

    This encoder applies a sequence of SelfAttnBlockAPI blocks.
    """

    def __init__(
            self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            block_style_list=['attn-deform'], combine_style='concat', k=10, n_group=2
    ):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                SelfAttnBlockAPI(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    block_style=block_style_list[i],
                    combine_style=combine_style,
                    k=k,
                    n_group=n_group
                )
            )

    def forward(self, x, pos):
        # If positional information is provided, compute k-NN indices
        if pos is not None:
            idx = knn_point(self.k, pos, pos)
        else:
            idx = None
        for _, block in enumerate(self.blocks):
            x = block(x, pos, idx=idx)
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder without hierarchical structure

    This decoder applies a sequence of CrossAttnBlockAPI blocks.
    """

    def __init__(
            self, embed_dim=256, depth=4, num_heads=4, mlp_ratio=4., qkv_bias=False, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2
    ):
        super().__init__()
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                CrossAttnBlockAPI(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    init_values=init_values,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    self_attn_block_style=self_attn_block_style_list[i],
                    self_attn_combine_style=self_attn_combine_style,
                    cross_attn_block_style=cross_attn_block_style_list[i],
                    cross_attn_combine_style=cross_attn_combine_style,
                    k=k,
                    n_group=n_group
                )
            )

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        if denoise_length is None:
            self_attn_idx = knn_point(self.k, q_pos, q_pos)
        else:
            self_attn_idx = None
        if q_pos is None:
            cross_attn_idx = None
        else:
            cross_attn_idx = knn_point(self.k, v_pos, q_pos)
        for _, block in enumerate(self.blocks):
            q = block(
                q, v, q_pos, v_pos,
                self_attn_idx=self_attn_idx,
                cross_attn_idx=cross_attn_idx,
                denoise_length=denoise_length
            )
        return q


class PointTransformerEncoder(nn.Module):
    """
    Vision Transformer for point cloud encoder

    A PyTorch implementation inspired by:
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    """

    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            block_style_list=['attn-deform'], combine_style='concat',
            k=10, n_group=2
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim  # for consistency with other models
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        assert len(block_style_list) == depth
        self.blocks = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_style_list=block_style_list,
            combine_style=combine_style,
            k=k,
            n_group=n_group
        )
        self.norm = norm_layer(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos=None):
        x = self.blocks(x, pos)
        return x


class PointTransformerDecoder(nn.Module):
    """
    Vision Transformer for point cloud decoder

    A PyTorch implementation inspired by:
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
    """

    def __init__(
            self, embed_dim=256, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=True, init_values=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=None, act_layer=None,
            self_attn_block_style_list=['attn-deform'], self_attn_combine_style='concat',
            cross_attn_block_style_list=['attn-deform'], cross_attn_combine_style='concat',
            k=10, n_group=2
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.num_features = self.embed_dim = embed_dim
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        assert len(self_attn_block_style_list) == len(cross_attn_block_style_list) == depth
        self.blocks = TransformerDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dpr,
            norm_layer=norm_layer,
            act_layer=act_layer,
            self_attn_block_style_list=self_attn_block_style_list,
            self_attn_combine_style=self_attn_combine_style,
            cross_attn_block_style_list=cross_attn_block_style_list,
            cross_attn_combine_style=cross_attn_combine_style,
            k=k,
            n_group=n_group
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, q, v, q_pos, v_pos, denoise_length=None):
        q = self.blocks(q, v, q_pos, v_pos, denoise_length=denoise_length)
        return q


# Entry wrappers to build encoder/decoder from configuration
class PointTransformerEncoderEntry(PointTransformerEncoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))


class PointTransformerDecoderEntry(PointTransformerDecoder):
    def __init__(self, config, **kwargs):
        super().__init__(**dict(config))

class DGCNN_Grouper(nn.Module):
    """
    Dynamic Graph CNN Grouper

    Groups points using DGCNN and applies several convolutional layers to extract features.
    """

    def __init__(self, k=16):
        super().__init__()
        # K must be 16
        self.k = k
        self.input_trans = nn.Conv1d(3, 8, 1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.num_features = 128

    @staticmethod
    def fps_downsample(coor, x, normal, plane_idx, num_group):
        """
        Farthest point sampling downsample

        Args:
            coor: Coordinates tensor (B, C, N)
            x: Feature tensor
            normal: Normal vectors tensor
            plane_idx: Plane index tensor
            num_group: Number of groups

        Returns:
            new_coor, new_normal, new_x, new_plane_idx after sampling
        """
        xyz = coor.transpose(1, 2).contiguous()  # B, N, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)
        combined_x = torch.cat([coor, normal, plane_idx, x], dim=1)
        new_combined_x = pointnet2_utils.gather_operation(combined_x, fps_idx)
        new_coor = new_combined_x[:, :3, :]
        new_normal = new_combined_x[:, 3:6, :]
        new_plane_idx = new_combined_x[:, 6, :].unsqueeze(1)
        new_x = new_combined_x[:, 7:, :]
        return new_coor, new_normal, new_x, new_plane_idx

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):
        """
        Compute graph features using k-NN

        Args:
            coor_q: Query coordinates tensor
            x_q: Query feature tensor
            coor_k: Key coordinates tensor
            x_k: Key feature tensor

        Returns:
            Graph feature tensor
        """
        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            idx = knn_point(
                k,
                coor_k.transpose(-1, -2).contiguous(),
                coor_q.transpose(-1, -2).contiguous()
            )
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, x, num):
        """
        Forward pass for grouping

        Args:
            x: Input tensor with shape (B, N, 7) where 7 corresponds to x,y,z,nx,ny,nz,plane_id
            num: List with grouping parameters

        Returns:
            coor, f, normal, plane_idx after grouping
        """
        assert x.shape[-1] == 7
        batch_size, num_points, _ = x.size()
        x = x.transpose(-1, -2).contiguous()
        coor, normal, plane_idx = x[:, :3, :], x[:, 3:6, :], x[:, -1, :].unsqueeze(1)
        f = self.input_trans(coor)
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer1(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor_q, normal_q, f_q, plane_idx_q = self.fps_downsample(coor, f, normal, plane_idx, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer2(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor, normal, plane_idx = coor_q, normal_q, plane_idx_q
        f = self.get_graph_feature(coor, f, coor, f)
        f = self.layer3(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor_q, normal_q, f_q, plane_idx_q = self.fps_downsample(coor, f, normal, plane_idx, num[0])
        f = self.get_graph_feature(coor_q, f_q, coor, f)
        f = self.layer4(f)
        f = f.max(dim=-1, keepdim=False)[0]
        coor, normal, plane_idx = coor_q, normal_q, plane_idx_q
        coor = coor.transpose(-1, -2).contiguous()
        f = f.transpose(-1, -2).contiguous()
        normal = normal.transpose(-1, -2).contiguous()
        plane_idx = plane_idx.transpose(-1, -2).contiguous()
        return coor, f, normal, plane_idx


class Encoder(nn.Module):
    """
    Encoder module for extracting global features from point groups

    Processes point groups with sequential convolutions.
    """

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        """
        Forward pass for the Encoder

        Args:
            point_groups: Tensor with shape (B, G, N, 3)

        Returns:
            feature_global: Tensor with shape (B, G, C)
        """
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(bs, g, self.encoder_channel)


class SimpleEncoder(nn.Module):
    """
    Simple Encoder for point clouds using farthest point sampling and grouping
    """

    def __init__(self, k=32, embed_dims=128):
        super().__init__()
        self.embedding = Encoder(embed_dims)
        self.group_size = k
        self.num_features = embed_dims

    @staticmethod
    def fps(data, number):
        """
        Farthest point sampling

        Args:
            data: Tensor with shape (B, N, 7)
            number: Number of points to sample

        Returns:
            Sampled data tensor
        """
        fps_idx = pointnet2_utils.furthest_point_sample(data[:, :, :3], number)
        fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).contiguous()
        return fps_data

    def forward(self, xyz, n_group):
        """
        Forward pass for the SimpleEncoder

        Args:
            xyz: Input point cloud tensor (B, N, 7)
            n_group: Number of groups to sample

        Returns:
            center, features, normal, plane_idx tensors
        """
        if isinstance(n_group, list):
            n_group = n_group[-1]
        assert xyz.shape[-1] == 7
        batch_size, num_points, _ = xyz.shape
        coor = xyz[:, :, :3]
        xyz = self.fps(xyz, n_group)
        center, normal, plane_idx = xyz[:, :3, :], xyz[:, 3:6, :], xyz[:, -1, :].unsqueeze(1)
        idx = knn_point(self.group_size, coor, center)
        assert idx.size(1) == n_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = coor.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, n_group, self.group_size, 3).contiguous()
        assert neighborhood.size(1) == n_group
        assert neighborhood.size(2) == self.group_size
        features = self.embedding(neighborhood)
        return center, features, normal, plane_idx


class Fold(nn.Module):
    """
    Folding module to generate point clouds from latent features

    This module uses two folding layers to reconstruct a point cloud.
    """

    def __init__(self, in_channel, step, hidden_dim=512, freedom=3):
        super().__init__()
        self.in_channel = in_channel
        self.step = step
        self.freedom = freedom

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, self.freedom, 1)
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + self.freedom, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, self.freedom, 1)
        )

    def forward(self, x):
        """
        Forward pass for the folding module

        Args:
            x: Input latent feature tensor

        Returns:
            Reconstructed point cloud offsets
        """
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)
        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)
        return fd2


class SimpleRebuildFCLayer(nn.Module):
    """
    Simple Fully Connected Layer for Rebuilding Point Clouds

    Applies an MLP on concatenated global and token features.
    """

    def __init__(self, input_dims, step, hidden_dim=512, freedom=3):
        super().__init__()
        self.input_dims = input_dims
        self.step = step
        self.freedom = freedom
        self.layer = MLP(self.input_dims, hidden_dim, step * self.freedom)

    def forward(self, rec_feature):
        """
        Forward pass for rebuilding

        Args:
            rec_feature: Reconstruction feature tensor with shape (B, N, C)

        Returns:
            Rebuilt point cloud tensor with shape (B, N, step, freedom)
        """
        batch_size = rec_feature.size(0)
        g_feature = rec_feature.max(1)[0]
        token_feature = rec_feature
        patch_feature = torch.cat([
            g_feature.unsqueeze(1).expand(-1, token_feature.size(1), -1),
            token_feature
        ], dim=-1)
        rebuild_pc = self.layer(patch_feature).reshape(batch_size, -1, self.step, self.freedom)
        assert rebuild_pc.size(1) == rec_feature.size(1)
        return rebuild_pc


class PCTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder = config.encoder
        decoder = config.decoder
        self.num_centers = getattr(config, 'num_centers', [512, 128])
        self.num_planes = getattr(config, 'num_planes', 20)
        self.group_k = getattr(config, 'group_k', 32)
        self.query_ranking = getattr(config, 'query_ranking', False)
        self.encoder_type = config.encoder_type
        self.query_type = config.query_type
        in_chans = 3
        self.num_queries = config.num_queries
        query_num = config.num_queries
        global_feature_dim = config.global_feature_dim
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k=self.group_k)
            self.plane_mlp = nn.Sequential(
                nn.Linear(encoder.embed_dim * 2, encoder.embed_dim),
                nn.GELU()
            )
        else:
            self.grouper = SimpleEncoder(k=self.group_k, num_planes=self.num_planes)
        self.pos_embed = nn.Sequential(
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, encoder.embed_dim)
        )
        self.plane_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, encoder.embed_dim)
        )
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, encoder.embed_dim)
        )
        self.encoder = PointTransformerEncoderEntry(encoder)
        self.increase_dim = nn.Sequential(
            nn.Linear(encoder.embed_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, global_feature_dim))
        if self.query_type == 'dynamic':
            self.plane_pred_coarse = nn.Sequential(
                nn.Linear(global_feature_dim, 1024),
                nn.GELU(),
                nn.Linear(1024, 3 * query_num))
            self.mlp_query = nn.Sequential(
                nn.Linear(global_feature_dim + 3, 1024),
                nn.GELU(),
                nn.Linear(1024, 1024),
                nn.GELU(),
                nn.Linear(1024, decoder.embed_dim))
            self.plane_pred = nn.Sequential(
                nn.Linear(decoder.embed_dim, decoder.embed_dim // 2),
                nn.GELU(),
                nn.Linear(decoder.embed_dim // 2, 3)
            )
        else:
            self.mlp_query = nn.Embedding(query_num, decoder.embed_dim)
            self.plane_pred = nn.Sequential(
                nn.Linear(decoder.embed_dim, decoder.embed_dim // 2),
                nn.GELU(),
                nn.Linear(decoder.embed_dim // 2, 3)
            )
        # assert decoder.embed_dim == encoder.embed_dim
        if decoder.embed_dim == encoder.embed_dim:
            self.mem_link = nn.Identity()
        else:
            self.mem_link = nn.Linear(encoder.embed_dim, decoder.embed_dim)
        self.decoder = PointTransformerDecoderEntry(decoder)
        if self.query_ranking:
            self.plane_mlp2 = nn.Sequential(nn.Linear(encoder.embed_dim, encoder.embed_dim),
                                            nn.GELU())
            self.query_ranking = nn.Sequential(
                nn.Linear(decoder.embed_dim, 256),
                nn.GELU(),
                nn.Linear(256, 256),
                nn.GELU(),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, xyz):
        bs, _, _ = xyz.size()
        coor, f, normal, batch = self.grouper(xyz, self.num_centers)
        pe = self.pos_embed(coor)
        x = self.input_proj(f)
        x = self.encoder(x + pe, coor)
        # from point proxy to plane proxy
        normal_embed = self.plane_embed(normal)
        x = torch.cat([x, normal_embed], dim=-1)
        plane_encoder = torch.zeros(bs, self.num_planes, x.size(-1)).cuda()
        for i in range(bs):
            unique_batch = torch.unique(batch[i])
            for j, ub in enumerate(unique_batch):
                plane_encoder[i][j] = x[i][(batch[i] == ub).squeeze()].sum(dim=0)
        x = self.plane_mlp(plane_encoder)
        global_feature = self.increase_dim(x)  # B 1024 N
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024
        mem = self.mem_link(x)
        f_plane = torch.zeros(bs, self.num_planes, normal_embed.size(-1)).cuda()
        for i in range(bs):
            unique_batch = torch.unique(batch[i])
            f_plane[i][:unique_batch.shape[0]] = x[i][:unique_batch.shape[0]]
        if self.query_type == 'dynamic':
            plane = self.plane_pred_coarse(global_feature).reshape(bs, -1, 3)
            q = self.mlp_query(
                torch.cat([
                    global_feature.unsqueeze(1).expand(-1, plane.size(1), -1),
                    plane], dim=-1))
            if self.query_ranking:
                f_plane = self.plane_mlp2(f_plane)
                q = torch.cat([q, f_plane], dim=1)
                query_ranking = self.query_ranking(q)
                idx = torch.argsort(query_ranking, dim=1)
                q = torch.gather(q, 1, idx[:, :self.num_queries].expand(-1, -1, q.size(-1)))
            q = self.decoder(q=q, v=mem, q_pos=None, v_pos=None, denoise_length=0)
            plane = self.plane_pred(q).reshape(bs, -1, 3)
        elif self.query_type == 'static':
            q = self.mlp_query.weight
            q = q.unsqueeze(0).expand(bs, -1, -1)
            q = self.decoder(q=q, v=mem, q_pos=None, v_pos=None, denoise_length=0)
            plane = self.plane_pred(q).reshape(bs, -1, 3)
        return q, plane

class HierarchicalPointEncoder(nn.Module):
    """
    层次化点云编码器，结合Point-E和Point-VQVAE思想
    """
    def __init__(self, config):
        super().__init__()
        self.num_levels = getattr(config, 'num_levels', 3)
        self.embed_dim = config.encoder.embed_dim
        self.codebook_size = getattr(config, 'codebook_size', 8192)
        self.encoder_type = config.encoder_type
        
        # 基础编码器 (复用PaCo的组件)
        if self.encoder_type == 'graph':
            self.grouper = DGCNN_Grouper(k=config.group_k)
        else:
            self.grouper = SimpleEncoder(k=config.group_k, embed_dims=128)
        
        # 位置编码
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.embed_dim)
        )
        
        # 特征投影
        self.input_proj = nn.Sequential(
            nn.Linear(self.grouper.num_features, 512),
            nn.GELU(),
            nn.Linear(512, self.embed_dim)
        )
        
        # Transformer编码器
        self.encoder = PointTransformerEncoderEntry(config.encoder)
        
        # 层次化特征金字塔
        self.pyramid_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self.embed_dim, self.embed_dim * 2, 1),
                nn.GroupNorm(8, self.embed_dim * 2),
                nn.GELU(),
                nn.Conv1d(self.embed_dim * 2, self.embed_dim, 1),
                nn.GroupNorm(8, self.embed_dim),
                nn.GELU()
            ) for _ in range(self.num_levels)
        ])
        
        # VQ-VAE量化层 - 每个层次不同的codebook大小
        self.vq_layers = nn.ModuleList([
            VectorQuantizer(
                n_embed=max(512, self.codebook_size // (2**i)),  # 递减的codebook大小
                embed_dim=self.embed_dim,
                beta=0.25
            ) for i in range(self.num_levels)
        ])
        
        # 层次间融合
        self.level_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(self.num_levels - 1)
        ])

    def forward(self, xyz) -> List[Dict]:
        """
        前向传播
        
        Args:
            xyz: 输入点云 [B, N, 7]
        
        Returns:
            List of hierarchical codes
        """
        bs = xyz.size(0)
        
        # 基础特征提取
        coor, f, normal, batch = self.grouper(xyz, self.num_centers if hasattr(self, 'num_centers') else [128, 128])
        pe = self.pos_embed(coor)
        x = self.input_proj(f)
        
        # Transformer编码
        x = self.encoder(x + pe, coor)  # [B, M, C]
        
        hierarchical_codes = []
        base_features = x
        
        # 生成多层次编码
        for level in range(self.num_levels):
            # 计算当前层次的分辨率
            current_resolution = base_features.size(1) // (2 ** level)
            current_resolution = max(current_resolution, 8)  # 最小分辨率
            
            # 下采样到当前层次分辨率
            if current_resolution == base_features.size(1):
                level_features = base_features
            else:
                # 使用自适应平均池化下采样
                level_features = F.adaptive_avg_pool1d(
                    base_features.transpose(1, 2), 
                    current_resolution
                ).transpose(1, 2)
            
            # 层次化特征处理
            level_features_conv = self.pyramid_encoders[level](level_features.transpose(1, 2))
            level_features_processed = level_features_conv.transpose(1, 2)
            
            # 如果不是第一层，融合上一层的信息
            if level > 0 and len(hierarchical_codes) > 0:
                prev_features = hierarchical_codes[-1]['features']
                # 上采样前一层特征到当前分辨率
                if prev_features.size(1) != current_resolution:
                    prev_upsampled = F.interpolate(
                        prev_features.transpose(1, 2),
                        size=current_resolution,
                        mode='linear',
                        align_corners=False
                    ).transpose(1, 2)
                else:
                    prev_upsampled = prev_features
                
                # 特征融合
                fused_input = torch.cat([level_features_processed, prev_upsampled], dim=-1)
                level_features_processed = self.level_fusion[level-1](fused_input)
            
            # VQ量化
            quantized, vq_loss, encoding_indices = self.vq_layers[level](level_features_processed)
            
            hierarchical_codes.append({
                'features': quantized,
                'indices': encoding_indices,
                'loss': vq_loss,
                'level': level,
                'resolution': current_resolution,
                'raw_features': level_features_processed
            })
        
        return hierarchical_codes


class HierarchicalDecoder(nn.Module):
    """
    层次化解码器，从多层次编码重建特征
    """
    def __init__(self, embed_dim: int, num_levels: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        
        # 上采样网络
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(embed_dim, embed_dim, 4, stride=2, padding=1),
                nn.GroupNorm(8, embed_dim),
                nn.GELU(),
                nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
                nn.GroupNorm(8, embed_dim),
                nn.GELU()
            ) for _ in range(num_levels - 1)
        ])
        
        # 特征融合
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim * 2, embed_dim, 1),
                nn.GroupNorm(8, embed_dim),
                nn.GELU(),
                nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
                nn.GroupNorm(8, embed_dim),
                nn.GELU()
            ) for _ in range(num_levels - 1)
        ])
        
        # 最终输出层
        self.output_layer = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim * 2, 1),
            nn.GroupNorm(8, embed_dim * 2),
            nn.GELU(),
            nn.Conv1d(embed_dim * 2, embed_dim, 1)
        )

    def forward(self, hierarchical_codes: List[Dict]) -> torch.Tensor:
        """
        从层次化编码重建特征
        
        Args:
            hierarchical_codes: 层次化编码列表
        
        Returns:
            重建的特征 [B, target_resolution, C]
        """
        # 从最粗糙的层次开始
        current_features = hierarchical_codes[-1]['features']  # 最后一层是最粗糙的
        target_resolution = hierarchical_codes[0]['resolution']  # 第一层是最精细的
        
        # 逐层上采样和融合
        for i in range(self.num_levels - 2, -1, -1):  # 从倒数第二层开始到第一层
            # 上采样当前特征
            current_features_conv = current_features.transpose(1, 2)  # [B, C, N]
            
            if i < len(self.upsample_layers):
                upsampled = self.upsample_layers[i](current_features_conv)
            else:
                upsampled = current_features_conv
            
            # 调整到目标分辨率
            target_res = hierarchical_codes[i]['resolution']
            if upsampled.size(-1) != target_res:
                upsampled = F.interpolate(
                    upsampled, 
                    size=target_res, 
                    mode='linear', 
                    align_corners=False
                )
            
            # 获取当前层的特征并融合
            level_features = hierarchical_codes[i]['features'].transpose(1, 2)
            
            # 确保尺寸匹配
            if upsampled.size(-1) != level_features.size(-1):
                min_size = min(upsampled.size(-1), level_features.size(-1))
                upsampled = F.interpolate(upsampled, size=min_size, mode='linear', align_corners=False)
                level_features = F.interpolate(level_features, size=min_size, mode='linear', align_corners=False)
            
            # 特征融合
            fused = torch.cat([upsampled, level_features], dim=1)
            if i < len(self.fusion_layers):
                current_features = self.fusion_layers[i](fused).transpose(1, 2)
            else:
                current_features = fused.transpose(1, 2)
        
        # 最终处理
        output = self.output_layer(current_features.transpose(1, 2)).transpose(1, 2)
        
        # 确保输出分辨率正确
        if output.size(1) != target_resolution:
            output = F.interpolate(
                output.transpose(1, 2), 
                size=target_resolution, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        return output

class HierarchicalPCTransformer(PCTransformer):
    """
    继承PCTransformer并添加层次化编码支持
    """
    def __init__(self, config):
        super().__init__(config)
        
        # 添加层次化编码器
        self.hierarchical_encoder = HierarchicalPointEncoder(config)
        self.num_levels = getattr(config, 'num_levels', 3)
        self.use_hierarchical = getattr(config, 'use_hierarchical', True)
        
        # 层次化解码器
        if self.use_hierarchical:
            self.hierarchical_decoder = HierarchicalDecoder(
                self.encoder.embed_dim, 
                self.num_levels
            )

    def forward(self, xyz):
        """
        前向传播，支持层次化特征提取
        
        Args:
            xyz: 输入点云 [B, N, 7]
        
        Returns:
            q: 查询特征 [B, M, C]
            plane: 平面参数 [B, M, 3]
            hierarchical_codes: 层次化编码 (如果启用)
        """
        if self.use_hierarchical:
            # 使用层次化编码器
            hierarchical_codes = self.hierarchical_encoder(xyz)
            
            # 从层次化编码重建特征
            reconstructed_features = self.hierarchical_decoder(hierarchical_codes)
            
            # 后续处理使用重建的特征
            bs = xyz.size(0)
            
            # 平面嵌入处理 (简化版)
            if hasattr(self, 'plane_embed'):
                # 使用grouper获取normal信息
                _, _, normal, batch = self.grouper(xyz, self.num_centers)
                normal_embed = self.plane_embed(normal)
                x = torch.cat([reconstructed_features, normal_embed], dim=-1)
            else:
                x = reconstructed_features
            
            # 平面处理
            if hasattr(self, 'plane_mlp'):
                plane_encoder = torch.zeros(bs, self.num_planes, x.size(-1)).to(x.device)
                for i in range(bs):
                    if hasattr(self, 'grouper'):
                        _, _, _, batch = self.grouper(xyz, self.num_centers)
                        unique_batch = torch.unique(batch[i])
                        for j, ub in enumerate(unique_batch):
                            if j < self.num_planes:
                                plane_encoder[i][j] = x[i][(batch[i] == ub).squeeze()].sum(dim=0)
                x = self.plane_mlp(plane_encoder)
            
            # 全局特征
            global_feature = self.increase_dim(x)  # B, N, 1024
            global_feature = torch.max(global_feature, dim=1)[0]  # B, 1024
            mem = self.mem_link(x)
            
            # 查询生成
            if self.query_type == 'dynamic':
                plane = self.plane_pred_coarse(global_feature).reshape(bs, -1, 3)
                q = self.mlp_query(
                    torch.cat([
                        global_feature.unsqueeze(1).expand(-1, plane.size(1), -1),
                        plane
                    ], dim=-1)
                )
                
                if self.query_ranking:
                    # 简化的query ranking
                    q = self.decoder(q=q, v=mem, q_pos=None, v_pos=None, denoise_length=0)
                    
                plane = self.plane_pred(q).reshape(bs, -1, 3)
            else:
                q = self.mlp_query.weight.unsqueeze(0).expand(bs, -1, -1)
                q = self.decoder(q=q, v=mem, q_pos=None, v_pos=None, denoise_length=0)
                plane = self.plane_pred(q).reshape(bs, -1, 3)
            
            return q, plane, hierarchical_codes
        else:
            # 使用原始方法
            q, plane = super().forward(xyz)
            return q, plane, None


@MODELS.register_module()
class HierarchicalDiffusionPaCo(nn.Module):
    """
    Hierarchical Diffusion PaCo Model
    保持与原始PaCo相同的输入输出接口
    """
    
    def __init__(self, config, **kwargs):
        super().__init__()
        # 保持原始PaCo的所有配置
        self.trans_dim = config.decoder.embed_dim
        self.num_queries = config.num_queries
        self.num_points = getattr(config, 'num_points', None)
        self.decoder_type = config.decoder_type
        self.fold_step = 8
        self.repulsion = config.repulsion
        
        # 层次化和扩散相关配置
        self.num_levels = getattr(config, 'num_levels', 3)
        self.num_timesteps = getattr(config, 'num_timesteps', 1000)
        self.codebook_size = getattr(config, 'codebook_size', 8192)
        self.use_diffusion = getattr(config, 'use_diffusion', True)
        self.training_mode = getattr(config, 'training_mode', 'joint')
        self.diffusion_weight = getattr(config, 'diffusion_weight', 1.0)
        self.vq_weight = getattr(config, 'vq_weight', 0.25)
        
        # 修改配置以启用层次化
        config.use_hierarchical = True
        config.num_levels = self.num_levels
        config.codebook_size = self.codebook_size
        
        # 基础PaCo组件 (使用层次化版本)
        self.base_model = HierarchicalPCTransformer(config)
        
        # 扩散模型组件
        if self.use_diffusion:
            self.noise_scheduler = NoiseScheduler(self.num_timesteps)
            
            # 层次化扩散UNet
            self.diffusion_models = nn.ModuleList([
                DiffusionUNet(
                    in_channels=self.trans_dim,
                    model_channels=self.trans_dim // 2,
                    out_channels=self.trans_dim,
                    condition_channels=self.trans_dim,
                    num_heads=8
                ) for _ in range(self.num_levels)
            ])
            
            # 跨层次注意力
            self.cross_level_attention = nn.ModuleList([
                CrossLevelAttention(self.trans_dim, num_heads=8)
                for _ in range(self.num_levels - 1)
            ])
        
        # 保持原始PaCo的解码头
        if self.decoder_type == 'fold':
            self.factor = self.fold_step ** 2
            self.decode_head = Fold(self.trans_dim, step=self.fold_step, hidden_dim=256, freedom=2)
        else:
            if self.num_points is not None:
                self.factor = self.num_points // self.num_queries
                self.decode_head = SimpleRebuildFCLayer(
                    self.trans_dim * 2,
                    step=self.num_points // self.num_queries,
                    freedom=2
                )
            else:
                self.factor = self.fold_step ** 2
                self.decode_head = SimpleRebuildFCLayer(
                    self.trans_dim * 2,
                    step=self.fold_step ** 2,
                    freedom=2
                )

        # 保持原始的其他组件
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        hidden_dim = 256

        self.rebuild_map = nn.Sequential(
            nn.Conv1d(self.trans_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_dim // 2, 2)
        
        # 存储中间结果用于损失计算
        self._hierarchical_codes = None
        self._predicted_noise = None
        self._true_noise = None
        self._timesteps = None

    def hierarchical_diffusion_forward(self, hierarchical_codes: List[Dict], timesteps: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        层次化扩散前向过程
        
        Args:
            hierarchical_codes: 层次化编码
            timesteps: 时间步 [B]
        
        Returns:
            predicted_noise: 预测的噪声
            true_noise: 真实添加的噪声
        """
        if not self.use_diffusion:
            return [code['features'] for code in hierarchical_codes], []
        
        predicted_noise = []
        true_noise = []
        prev_output = None
        
        for level, code in enumerate(hierarchical_codes):
            features = code['features']  # [B, N, C]
            B, N, C = features.shape
            
            # 生成噪声并添加到特征
            noise = torch.randn_like(features)
            true_noise.append(noise)
            
            if self.training:
                # 训练时：添加噪声
                noisy_features = self.noise_scheduler.add_noise(features, noise, timesteps)
            else:
                # 推理时：使用纯噪声或部分去噪的特征
                noisy_features = features + 0.1 * noise  # 轻微的噪声用于推理
            
            # 转换为 [B, C, N] 用于conv操作
            noisy_features = noisy_features.transpose(1, 2)
            
            # 准备条件
            if level == 0:
                # 第一层使用全局特征作为条件
                condition = features.mean(dim=1)  # [B, C]
            else:
                # 后续层使用前一层的输出
                if prev_output.size(-1) != N:
                    upsampled_prev = F.interpolate(prev_output, size=N, mode='linear', align_corners=False)
                else:
                    upsampled_prev = prev_output
                
                # 跨层注意力
                if level - 1 < len(self.cross_level_attention):
                    attended_input = self.cross_level_attention[level-1](noisy_features, upsampled_prev)
                    noisy_features = attended_input
                
                condition = upsampled_prev.mean(dim=-1)  # [B, C]
            
            # 扩散预测
            level_predicted_noise = self.diffusion_models[level](
                noisy_features, timesteps, condition
            )
            
            predicted_noise.append(level_predicted_noise)
            prev_output = level_predicted_noise
        
        return predicted_noise, true_noise

    def forward(self, xyz):
        """
        前向传播 - 保持与原始PaCo相同的接口
        
        Args:
            xyz: 输入点云 [B, N, 7]
        
        Returns:
            ret: (plane_params, rebuild_points) - 与原始PaCo相同
            class_prob: 分类概率 [B, M, 2] - 与原始PaCo相同
        """
        # 1. 基础特征提取 (使用层次化PCTransformer)
        result = self.base_model(xyz)
        if len(result) == 3:
            q, plane, hierarchical_codes = result
        else:
            q, plane = result
            hierarchical_codes = None
        
        B, M, C = q.shape
        
        # 2. 层次化编码 + 扩散处理 (仅在训练时或明确要求时)
        if self.use_diffusion and hierarchical_codes is not None:
            if self.training:
                # 训练时进行扩散
                timesteps = torch.randint(0, self.num_timesteps, (B,), device=q.device)
                predicted_noise, true_noise = self.hierarchical_diffusion_forward(hierarchical_codes, timesteps)
                
                # 存储中间结果用于损失计算
                self._hierarchical_codes = hierarchical_codes
                self._predicted_noise = predicted_noise
                self._true_noise = true_noise
                self._timesteps = timesteps
                
                # 使用去噪后的特征
                denoised_codes = []
                for i, (pred_noise, code) in enumerate(zip(predicted_noise, hierarchical_codes)):
                    # 简单的去噪：原特征 - 预测噪声
                    denoised = code['features'] - 0.1 * pred_noise.transpose(1, 2)
                    denoised_codes.append({
                        'features': denoised,
                        'resolution': code['resolution']
                    })
            else:
                # 验证时：使用轻量级扩散或固定时间步
                timesteps = torch.full((B,), self.num_timesteps // 2, device=q.device)  # 使用中等时间步
                predicted_noise, true_noise = self.hierarchical_diffusion_forward(hierarchical_codes, timesteps)
            
                # 存储结果但不用于反向传播
                self._hierarchical_codes = hierarchical_codes
                self._predicted_noise = predicted_noise
                self._true_noise = true_noise
                self._timesteps = timesteps
            
                # 使用轻微去噪的特征
                denoised_codes = []
                for i, (pred_noise, code) in enumerate(zip(predicted_noise, hierarchical_codes)):
                    denoised = code['features'] - 0.05 * pred_noise.transpose(1, 2)  # 更轻微的去噪
                    denoised_codes.append({
                        'features': denoised,
                        'resolution': code['resolution']
                    })
            # 重新解码特征
            if hasattr(self.base_model, 'hierarchical_decoder'):
                q_decoded = self.base_model.hierarchical_decoder(denoised_codes)
                if q_decoded.size(1) != M:
                    q = F.interpolate(
                        q_decoded.transpose(1, 2), 
                        size=M, 
                        mode='linear', 
                        align_corners=False
                    ).transpose(1, 2)
                else:
                    q = q_decoded
        
        # 确保 q 和 plane 的维度匹配
        B, M_q, C = q.shape
        B, M_plane, _ = plane.shape
        
        if M_q != M_plane:
            # 调整其中一个张量的尺寸以匹配另一个
            min_M = min(M_q, M_plane)
            q = q[:, :min_M, :]
            plane = plane[:, :min_M, :]
            M = min_M
        else:
            M = M_q
        
        # 3. 保持原始PaCo的后续处理流程
        global_feature = self.increase_dim(q.transpose(1, 2)).transpose(1, 2)  # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0]  # B 1024

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            plane
        ], dim=-1)  # B M (1028 + C)

        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))
            angle_point = self.decode_head(rebuild_feature).reshape(B, M, 2, -1)  # B M 2 S
            theta_point = angle_point[:, :, 0, :].unsqueeze(2)
            phi_point = angle_point[:, :, 1, :].unsqueeze(2)
        else:
            rebuild_feature = self.reduce_map(rebuild_feature)
            angle_point = self.decode_head(rebuild_feature)  # B M S 2
            theta_point = angle_point[:, :, :, 0]
            phi_point = angle_point[:, :, :, 1]

        rebuild_feature_for_classifier = self.rebuild_map(rebuild_feature.reshape(B * M, -1).unsqueeze(-1))

        # 保持原始的平面生成逻辑
        theta = plane[:, :, 0].unsqueeze(-1).expand(-1, -1, theta_point.size(-1) if theta_point.dim() == 3 else theta_point.size(-1))
        phi = plane[:, :, 1].unsqueeze(-1).expand(-1, -1, phi_point.size(-1) if phi_point.dim() == 3 else phi_point.size(-1))
        r = plane[:, :, 2].unsqueeze(-1).expand(-1, -1, theta_point.size(-1) if theta_point.dim() == 3 else theta_point.size(-1))
        
        N = torch.cos(phi_point - phi) * torch.sin(theta_point) * torch.sin(theta) + torch.cos(theta_point) * torch.cos(theta)
        N = torch.clamp(N, min=1e-6)
        r2 = r / N

        # Point cloud generation
        x_coord = (r2 * torch.sin(theta_point) * torch.cos(phi_point)).unsqueeze(-1)
        y_coord = (r2 * torch.sin(theta_point) * torch.sin(phi_point)).unsqueeze(-1)
        z_coord = (r2 * torch.cos(theta_point)).unsqueeze(-1)
        rebuild_points = torch.cat([x_coord, y_coord, z_coord], dim=-1)
        rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()
        rebuild_points = torch.clamp(rebuild_points, min=-1, max=1)

        # 转换平面参数格式
        a = torch.sin(plane[:, :, 0]) * torch.cos(plane[:, :, 1])
        b = torch.sin(plane[:, :, 0]) * torch.sin(plane[:, :, 1])
        c = torch.cos(plane[:, :, 0])
        d = -plane[:, :, 2]
        plane_params = torch.cat([a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1), d.unsqueeze(-1)], dim=-1)
        ret = (plane_params, rebuild_points)

        class_prob = self.classifier(rebuild_feature_for_classifier.squeeze(-1))  # B M 2
        class_prob = class_prob.reshape(B, M, -1)

        return ret, class_prob

    def get_diffusion_loss(self) -> Dict[str, torch.Tensor]:
        """
        计算扩散损失 (仅在训练时调用)
        """
        losses = {
            'diffusion_loss': torch.tensor(0.0, device=next(self.parameters()).device),
            'vq_loss': torch.tensor(0.0, device=next(self.parameters()).device)
        }
        
        if not hasattr(self, '_hierarchical_codes') or not self.use_diffusion or self._hierarchical_codes is None:
            return losses
        
        total_diffusion_loss = 0.0
        total_vq_loss = 0.0
        
        # VQ损失
        for code in self._hierarchical_codes:
            total_vq_loss += code['loss']
        
        # 扩散损失：预测噪声 vs 真实噪声
        if self._predicted_noise is not None and self._true_noise is not None:
            for predicted, true in zip(self._predicted_noise, self._true_noise):
                # predicted: [B, C, N], true: [B, N, C]
                true_transposed = true.transpose(1, 2)  # [B, C, N]
                diffusion_loss = F.mse_loss(predicted, true_transposed)
                total_diffusion_loss += diffusion_loss
        
        losses['diffusion_loss'] = total_diffusion_loss
        losses['vq_loss'] = total_vq_loss
        
        return losses

    def sample_with_diffusion(self, xyz, num_steps: int = 50):
        """
        使用完整扩散采样生成点云
        
        Args:
            xyz: 输入点云
            num_steps: 扩散采样步数
        
        Returns:
            与forward相同的输出格式
        """
        if not self.use_diffusion:
            return self.forward(xyz)
        
        with torch.no_grad():
            # 获取基础特征和层次化编码
            result = self.base_model(xyz)
            if len(result) == 3:
                q, plane, hierarchical_codes = result
            else:
                q, plane = result
                return self.forward(xyz)  # 降级到普通方法
            
            B, M, C = q.shape
            
            # 初始化噪声码本
            noise_codes = []
            for code in hierarchical_codes:
                noise = torch.randn_like(code['features'])
                noise_codes.append(noise)
            
            # 扩散采样循环
            for t in reversed(range(0, self.num_timesteps, self.num_timesteps // num_steps)):
                timestep = torch.tensor([t], device=q.device).expand(B)
                
                # 构造临时的层次化编码
                temp_codes = []
                for i, (noise, orig_code) in enumerate(zip(noise_codes, hierarchical_codes)):
                    temp_codes.append({
                        'features': noise,
                        'resolution': orig_code['resolution'],
                        'level': orig_code['level']
                    })
                
                # 预测噪声
                predicted_noise, _ = self.hierarchical_diffusion_forward(temp_codes, timestep)
                
                # 去噪步骤
                for i, (pred_noise, noise) in enumerate(zip(predicted_noise, noise_codes)):
                    # pred_noise: [B, C, N], noise: [B, N, C]
                    noise_transposed = noise.transpose(1, 2)  # [B, C, N]
                    
                    if t > 0:
                        denoised = self.noise_scheduler.step(pred_noise, t, noise_transposed)
                        noise_codes[i] = denoised.transpose(1, 2)  # Back to [B, N, C]
                    else:
                        # 最后一步，直接使用去噪结果
                        noise_codes[i] = (noise_transposed - pred_noise).transpose(1, 2)
            
            # 使用最终的去噪特征
            final_codes = []
            for noise, orig_code in zip(noise_codes, hierarchical_codes):
                final_codes.append({
                    'features': noise,
                    'resolution': orig_code['resolution']
                })
            
            # 重新解码
            if hasattr(self.base_model, 'hierarchical_decoder'):
                q_final = self.base_model.hierarchical_decoder(final_codes)
            else:
                q_final = q  # 降级
        
        # 使用去噪后的特征完成后续处理
        return self._forward_with_features(q_final, plane)

    def _forward_with_features(self, q, plane):
        """使用给定特征完成前向传播的后续部分"""
        B, M, C = q.shape
        
        global_feature = self.increase_dim(q.transpose(1, 2)).transpose(1, 2)
        global_feature = torch.max(global_feature, dim=1)[0]

        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            plane
        ], dim=-1)

        if self.decoder_type == 'fold':
            rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))
            angle_point = self.decode_head(rebuild_feature).reshape(B, M, 2, -1)
            theta_point = angle_point[:, :, 0, :].unsqueeze(2)
            phi_point = angle_point[:, :, 1, :].unsqueeze(2)
        else:
            rebuild_feature = self.reduce_map(rebuild_feature)
            angle_point = self.decode_head(rebuild_feature)
            theta_point = angle_point[:, :, :, 0]
            phi_point = angle_point[:, :, :, 1]

        rebuild_feature_classifier = self.rebuild_map(rebuild_feature.reshape(B * M, -1).unsqueeze(-1))

        # 平面生成
        theta = plane[:, :, 0].unsqueeze(-1).expand(-1, -1, theta_point.size(-1))
        phi = plane[:, :, 1].unsqueeze(-1).expand(-1, -1, phi_point.size(-1))
        r = plane[:, :, 2].unsqueeze(-1).expand(-1, -1, theta_point.size(-1))
        
        N = torch.cos(phi_point - phi) * torch.sin(theta_point) * torch.sin(theta) + torch.cos(theta_point) * torch.cos(theta)
        N = torch.clamp(N, min=1e-6)
        r2 = r / N

        x_coord = (r2 * torch.sin(theta_point) * torch.cos(phi_point)).unsqueeze(-1)
        y_coord = (r2 * torch.sin(theta_point) * torch.sin(phi_point)).unsqueeze(-1)
        z_coord = (r2 * torch.cos(theta_point)).unsqueeze(-1)
        rebuild_points = torch.cat([x_coord, y_coord, z_coord], dim=-1)
        rebuild_points = rebuild_points.reshape(B, -1, 3).contiguous()
        rebuild_points = torch.clamp(rebuild_points, min=-1, max=1)

        a = torch.sin(plane[:, :, 0]) * torch.cos(plane[:, :, 1])
        b = torch.sin(plane[:, :, 0]) * torch.sin(plane[:, :, 1])
        c = torch.cos(plane[:, :, 0])
        d = -plane[:, :, 2]
        plane_params = torch.cat([a.unsqueeze(-1), b.unsqueeze(-1), c.unsqueeze(-1), d.unsqueeze(-1)], dim=-1)
        ret = (plane_params, rebuild_points)

        class_prob = self.classifier(rebuild_feature_classifier.squeeze(-1))
        class_prob = class_prob.reshape(B, M, -1)

        return ret, class_prob

    def get_loss(self, config, ret, class_prob, gt, gt_index, plane, plan_index):
        """
           计算损失函数,集成原始PaCo损失和扩散损失
    
        Args:
            config: 配置对象，包含损失权重
            ret: 元组 (predicted_planes, reconstructed_points)
            class_prob: 分类概率张量 [B, M, 2]
            gt: 真实点云 [B, N, 3]
            gt_index: 真实点云索引 [B, N]
            plane: 预测平面参数 [B, M, 4]
            plan_index: 预测平面索引 [B, M]
    
        Returns:
            包含所有损失项的字典
        """
        predicted_planes, reconstructed_points = ret
        batch_size, _, _ = class_prob.size()
        device = reconstructed_points.device
        # 初始化损失字典（原始PaCo损失项）
        losses = {
            "plane_chamfer_loss": 0.0,
            "classification_loss": 0.0,
            "chamfer_norm1_loss": 0.0,
            "chamfer_norm2_loss": 0.0,
            "plane_normal_loss": 0.0,
            "repulsion_loss": 0.0
        }
        size_total = 0.0

        # 批处理损失计算（原始PaCo逻辑）
        for batch_idx in range(batch_size):
            # 获取唯一的真实平面索引
            unique_gt_indices = torch.unique(gt_index[batch_idx].int())
            unique_gt_indices = unique_gt_indices[unique_gt_indices != -1]  # 移除无效索引
            num_ground_truth_planes = unique_gt_indices.size(0)

            if num_ground_truth_planes == 0:
                continue
                # 计算真实点云集合
            ground_truth_pointclouds = [
                gt[batch_idx, (gt_index[batch_idx] == idx)].reshape(-1, 3)
                for idx in unique_gt_indices
            ]
            ground_truth_pointclouds = ground_truth_pointclouds * self.num_queries
            ground_truth_pointclouds = Pointclouds(ground_truth_pointclouds).to(device)

            # 计算重建点云集合
            start_indices = torch.arange(self.num_queries, device=device) * self.factor
            end_indices = start_indices + self.factor
            reconstructed_pointclouds = [
                reconstructed_points[batch_idx, start:end].reshape(-1, 3)
                for start, end in zip(start_indices, end_indices)
                ]
            reconstructed_pointclouds = [
                pointcloud for pointcloud in reconstructed_pointclouds for _ in range(num_ground_truth_planes)
            ]
            reconstructed_pointclouds = Pointclouds(reconstructed_pointclouds).to(device)
            # 计算平面Chamfer距离
            chamfer_distances, _ = chamfer_distance(
                ground_truth_pointclouds, reconstructed_pointclouds,
                point_reduction='mean', batch_reduction=None
            )
            plane_chamfer_distance = chamfer_distances.view(self.num_queries, num_ground_truth_planes)

            # 计算平面法向量损失
            gt_planes = plane[batch_idx, unique_gt_indices].float()
            pred_planes = predicted_planes[batch_idx]
            l2_loss = torch.sum((pred_planes[:, :3].unsqueeze(1) - gt_planes[:, :3].unsqueeze(0)) ** 2, dim=-1)
            cosine_loss = 1 - F.cosine_similarity(
                pred_planes[:, :3].unsqueeze(1), gt_planes[:, :3].unsqueeze(0), dim=-1
            )
            plane_normal_loss = (l2_loss + cosine_loss).view(self.num_queries, num_ground_truth_planes)

            # 计算分类损失
            classification_scores = class_prob[batch_idx]
            object_class_loss = F.cross_entropy(
                classification_scores,
                torch.zeros(self.num_queries, dtype=torch.long, device=device),
                size_average=False, reduce=False
            ).unsqueeze(-1).expand(-1, num_ground_truth_planes)
        
            non_object_class_loss = F.cross_entropy(
                classification_scores,
                torch.ones(self.num_queries, dtype=torch.long, device=device),
                size_average=False, reduce=False
            ).unsqueeze(-1).expand(-1, num_ground_truth_planes)

            # 计算排斥损失
            if hasattr(self, 'repulsion') and self.repulsion:
                reshaped_reconstructed_points = reconstructed_points[batch_idx].view(-1, self.factor, 3)
                neighbor_indices = knn_point(
                    self.repulsion.num_neighbors, reshaped_reconstructed_points, reshaped_reconstructed_points
                )[:, :, 1:].long()
            
                grouped_points = index_points(reshaped_reconstructed_points, neighbor_indices).transpose(2, 3).contiguous() - reshaped_reconstructed_points.unsqueeze(-1)
                distance_matrix = torch.sum(grouped_points ** 2, dim=2).clamp(min=self.repulsion.epsilon)
                weight_matrix = torch.exp(-distance_matrix / self.repulsion.kernel_bandwidth ** 2)
                repulsion_penalty = torch.mean(
                    (self.repulsion.radius - distance_matrix.sqrt()) * weight_matrix,
                    dim=(1, 2)
                ).clamp(min=0).unsqueeze(-1).expand(-1, num_ground_truth_planes)
            else:
                repulsion_penalty = torch.zeros(self.num_queries, num_ground_truth_planes, device=device)

            # 匈牙利分配算法进行匹配
            cost_matrix = (
                    object_class_loss +
                plane_chamfer_distance * config.plane_chamfer_loss_weight +
                repulsion_penalty * config.repulsion_loss_weight +
                plane_normal_loss * config.plane_normal_loss_weight
            )
        
            hungarian_assignment = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
            hungarian_assignment = [
                torch.tensor(a, dtype=torch.long, device=device) for a in hungarian_assignment
            ]

            # 提取匹配的损失
            matched_plane_chamfer_distance = plane_chamfer_distance[hungarian_assignment[0], hungarian_assignment[1]]
            matched_plane_normal_loss = plane_normal_loss[hungarian_assignment[0], hungarian_assignment[1]]
            matched_repulsion_penalty = repulsion_penalty[hungarian_assignment[0], 0]
            matched_reconstructed_points = reshaped_reconstructed_points[hungarian_assignment[0]].reshape(1, -1, 3)
            matched_object_class_loss = object_class_loss[hungarian_assignment[0], 0]

            # 计算未匹配的分类损失
            unmatched_indices = torch.tensor(
                list(set(range(self.num_queries)) - set(hungarian_assignment[0].tolist())),
                device=device
            )
            if len(unmatched_indices) > 0:
                unmatched_class_loss = non_object_class_loss[unmatched_indices, 0] * config.non_obj_class_loss_weight
                total_classification_loss = torch.cat([matched_object_class_loss, unmatched_class_loss])
            else:
                total_classification_loss = matched_object_class_loss

            # 计算细粒度Chamfer损失
            fine_chamfer_loss_1 = chamfer_distance(
                matched_reconstructed_points, gt[batch_idx].unsqueeze(0), norm=1
            )
            fine_chamfer_loss_2 = chamfer_distance(
                matched_reconstructed_points, gt[batch_idx].unsqueeze(0)
            )

            # 累积损失
            losses["plane_chamfer_loss"] += matched_plane_chamfer_distance.sum()
            losses["classification_loss"] += total_classification_loss.sum()
            losses["plane_normal_loss"] += matched_plane_normal_loss.sum()
            losses["repulsion_loss"] += matched_repulsion_penalty.sum()
            losses["chamfer_norm1_loss"] += fine_chamfer_loss_1[0]
            losses["chamfer_norm2_loss"] += fine_chamfer_loss_2[0]
            size_total += num_ground_truth_planes
         # 归一化损失
        for k in losses.keys():
            if k.startswith('chamfer'):
                losses[k] /= batch_size
            else:
                if size_total > 0:
                    losses[k] /= size_total
        
        # 添加扩散损失
        diffusion_losses = self.get_diffusion_loss()
        losses.update(diffusion_losses)

        # 计算总损失（原始PaCo损失 + 扩散损失）
        losses["total_loss"] = (
            losses["classification_loss"] +
            config.plane_chamfer_loss_weight * losses["plane_chamfer_loss"] +
            config.plane_normal_loss_weight * losses["plane_normal_loss"] +
            config.repulsion_loss_weight * losses["repulsion_loss"] +
            config.chamfer_norm2_loss_weight * losses["chamfer_norm2_loss"] +
            self.diffusion_weight * losses["diffusion_loss"] +
            self.vq_weight * losses["vq_loss"]
        )

        return losses