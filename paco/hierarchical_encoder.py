import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple
from .diffusion_components import VectorQuantizer
from .paco_pipeline import DGCNN_Grouper, SimpleEncoder, PointTransformerEncoderEntry


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