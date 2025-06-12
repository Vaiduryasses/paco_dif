import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List, Dict
import einops


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer for hierarchical encoding
    """
    def __init__(self, n_embed: int, embed_dim: int, beta: float = 0.25):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Input tensor [B, C, H] or [B, H, C]
        Returns:
            quantized: Quantized tensor
            loss: VQ loss
            encoding_indices: Indices used for quantization
        """
        # 确保输入张量的形状和维度处理
        original_shape = z.shape
        
        # 处理输入张量格式 - 统一为 [B, H, C] 格式
        if z.dim() == 3:
            if z.size(-1) != self.embed_dim:
                # 如果最后一个维度不是embed_dim，假设是 [B, C, H] 格式
                if z.size(1) == self.embed_dim:
                    z = z.permute(0, 2, 1)  # [B, C, H] -> [B, H, C]
                else:
                    raise ValueError(f"Input tensor shape {z.shape} is not compatible with embed_dim {self.embed_dim}")
        else:
            raise ValueError(f"Input tensor must be 3D, got {z.dim()}D")
        
        # 确保张量是连续的，然后reshape
        z = z.contiguous()
        z_flattened = z.reshape(-1, self.embed_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        
        # preserve gradients
        z_q = z + (z_q - z).detach()
        
        # 恢复原始形状
        if len(original_shape) == 3 and original_shape[1] == self.embed_dim:
            # 如果原始输入是 [B, C, H] 格式，转换回去
            z_q = z_q.permute(0, 2, 1)
        
        return z_q, loss, min_encoding_indices.view(z.shape[:-1])


class NoiseScheduler(nn.Module):
    """
    DDPM Noise Scheduler
    """
    def __init__(self, num_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        super().__init__()
        self.num_timesteps = num_timesteps
        
        # 将所有调度器参数注册为缓冲区，这样会自动管理设备
        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册为缓冲区（不参与梯度计算，但会跟随模型设备变化）
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        # For sampling
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior variance
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to samples"""
        # 确保 timesteps 在 CPU 上进行索引，然后移动到目标设备
        timesteps_cpu = timesteps.cpu()
        
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps_cpu].to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps_cpu].to(original_samples.device)
        
        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor) -> torch.Tensor:
        """Perform one denoising step"""
        t = timestep
        
        # Get parameters (already on correct device due to register_buffer)
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        
        # Compute the previous sample
        pred_original_sample = sqrt_recip_alphas_t * (sample - beta_t * model_output / sqrt_one_minus_alpha_cumprod_t)
        
        # Add noise if not the final step
        if t > 0:
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(sample)
            prev_sample = pred_original_sample + torch.sqrt(posterior_variance_t) * noise
        else:
            prev_sample = pred_original_sample
        
        return prev_sample


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block for UNet"""
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        
        self.block2 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb.unsqueeze(-1)
        
        h = self.block2(h)
        
        return h + self.res_conv(x)


class AttentionBlock(nn.Module):
    """Attention block for UNet"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
    def forward(self, x):
        B, C, N = x.shape
        
        # Normalize
        h = self.norm(x)
        
        # Transpose for attention: [B, C, N] -> [B, N, C]
        h = h.transpose(1, 2)
        
        # Self attention
        h, _ = self.attention(h, h, h)
        
        # Transpose back: [B, N, C] -> [B, C, N]
        h = h.transpose(1, 2)
        
        return x + h


class DiffusionUNet(nn.Module):
    """
    简化的 UNet，避免复杂的跳跃连接
    """
    def __init__(self, in_channels: int, model_channels: int, out_channels: int, 
                 condition_channels: int = None, num_heads: int = 8, num_res_blocks: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.condition_channels = condition_channels
        self.num_res_blocks = num_res_blocks
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Condition projection if needed
        if condition_channels is not None:
            self.condition_proj = nn.Linear(condition_channels, model_channels)
        
        # Initial projection
        self.input_proj = nn.Conv1d(in_channels, model_channels, 1)
        
        # 简化的编码器（不使用复杂跳跃连接）
        self.encoder = nn.ModuleList([
            ResidualBlock(model_channels, model_channels, time_embed_dim),
            AttentionBlock(model_channels, num_heads),
            ResidualBlock(model_channels, model_channels * 2, time_embed_dim),
            nn.AvgPool1d(2),
            ResidualBlock(model_channels * 2, model_channels * 2, time_embed_dim),
            AttentionBlock(model_channels * 2, num_heads),
        ])
        
        # Middle block
        self.middle = nn.ModuleList([
            ResidualBlock(model_channels * 2, model_channels * 4, time_embed_dim),
            AttentionBlock(model_channels * 4, num_heads),
            ResidualBlock(model_channels * 4, model_channels * 2, time_embed_dim),
        ])
        
        # 简化的解码器
        self.decoder = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            ResidualBlock(model_channels * 2, model_channels, time_embed_dim),
            AttentionBlock(model_channels, num_heads),
            ResidualBlock(model_channels, model_channels, time_embed_dim),
        ])
        
        # Output
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, condition: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, N]
            timestep: Timestep tensor [B]
            condition: Condition tensor [B, N, C] or [B, C]
        """
        B, C, N = x.shape
        
        # Time embedding
        t_emb = self.time_embed(timestep)  # [B, time_embed_dim]
        
        # Input projection
        h = self.input_proj(x)  # [B, model_channels, N]
        
        # Add condition if provided
        if condition is not None and self.condition_proj is not None:
            if condition.dim() == 2:  # [B, C]
                cond = self.condition_proj(condition).unsqueeze(-1).expand(-1, -1, N)
            else:  # [B, N, C]
                cond = self.condition_proj(condition).transpose(1, 2)  # [B, C, N]
            h = h + cond
        
        # Encoder
        for layer in self.encoder:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Decoder
        for layer in self.decoder:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
        
        # Output
        output = self.output_proj(h)
        
        return output


class CrossLevelAttention(nn.Module):
    """
    Cross-level attention for hierarchical fusion
    """
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, current_level: torch.Tensor, prev_level: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current_level: [B, C, N]
            prev_level: [B, C, M]
        """
        B, C, N = current_level.shape
        _, _, M = prev_level.shape
        
        # Transpose for attention
        curr = current_level.transpose(1, 2)  # [B, N, C]
        prev = prev_level.transpose(1, 2)     # [B, M, C]
        
        # Interpolate prev to match current size if needed
        if M != N:
            prev = F.interpolate(prev_level, size=N, mode='linear', align_corners=False).transpose(1, 2)
        
        # Cross attention
        attended, _ = self.attention(self.norm1(curr), self.norm2(prev), self.norm2(prev))
        attended = attended + curr
        
        # Combine with original
        combined = torch.cat([curr, attended], dim=-1)
        output = self.proj(combined)
        
        return output.transpose(1, 2)  # Back to [B, C, N]


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)