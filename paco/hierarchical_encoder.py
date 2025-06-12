import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from .diffusion_components import VectorQuantizer
from .paco_pipeline import  SimpleEncoder, PointTransformerEncoderEntry


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