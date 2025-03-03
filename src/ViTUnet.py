from src.NewUnet import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchvision import transforms


# ----------------------
# ViT 模块（用于增强全局注意力）
# ----------------------
# 这里我们实现一个简单的 ViT 模块：
class ViTModule(nn.Module):
    def __init__(self, in_channels, img_size, patch_size, embed_dim, num_layers, num_heads):
        """
        参数说明：
         - in_channels: 输入特征图通道数（来自 bottleneck）
         - img_size: 输入特征图的尺寸（假设为正方形，如 32）
         - patch_size: Patch 大小（必须整除 img_size）
         - embed_dim: Transformer 嵌入维度
         - num_layers: Transformer Encoder 层数
         - num_heads: 多头注意力头数
        """
        super(ViTModule, self).__init__()
        assert img_size % patch_size == 0, "img_size 必须是 patch_size 的整数倍"
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, in_channels)

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        # Transformer Encoder 期望输入 (seq_len, batch, embed_dim)
        x = x.transpose(0, 1)  # (num_patches, B, embed_dim)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)  # (B, num_patches, embed_dim)
        x = self.proj(x)       # (B, num_patches, in_channels)
        # 恢复回空间维度
        h = H // self.patch_size
        w = W // self.patch_size
        x = x.transpose(1, 2).reshape(B, C, h, w)
        # 如果需要，可以上采样回原始尺寸
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return x


class PreViT(nn.Module):
    def __init__(self, in_channels, vit_img_size, vit_patch_size, vit_embed_dim,
                 vit_num_layers, vit_num_heads):
        super(PreViT, self).__init__()
        self.vit = ViTModule(
            in_channels=in_channels,
            img_size=vit_img_size,
            patch_size=vit_patch_size,
            embed_dim=vit_embed_dim,
            num_layers=vit_num_layers,
            num_heads=vit_num_heads
        )
        # 如果需要，可以用一个卷积层将 ViT 输出的通道映射到原始通道数
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.vit(x)
        x = self.conv(x)
        return x


class ViTUnet(UNet):
    def __init__(self,
                 down_chs: Tuple[int, ...] = (6, 64, 128, 256),
                 up_chs: Tuple[int, ...] = (256, 128, 64),
                 num_class: int = 1,
                 retain_dim: bool = False,
                 out_sz: Tuple[int, int] = (572, 572),
                 kernel_size: int = 3,
                 # previt_img_size: int = 572,  # 根据输入图像尺寸设置
                 # previt_patch_size: int = 4,  # 根据情况选择 patch 大小
                 vit_img_size: int = 32,  # 这里需根据 bottleneck 输出特征图尺寸来设置
                 vit_patch_size: int = 4,  # 通常取较小值
                 vit_embed_dim: int = 256,
                 vit_num_layers: int = 6,
                 vit_num_heads: int = 8):
        """
        修改后的复杂版 UNet，其输入参数与简单版完全一致：
         - enc_chs: Encoder 各层的通道数（默认 (3, 64, 128)）
         - dec_chs: Decoder 各层的输出通道数（默认 (128, 64)）
         - num_class: 输出类别数（默认 1）
         - retain_dim: 是否通过插值调整输出尺寸到 out_sz（默认 False）
         - out_sz: 当 retain_dim 为 True 时的输出尺寸（默认 (572,572)）
         - kernel_size: 卷积核大小（默认 3）
         - binary_class: 是否为二分类问题（默认 True）
        """
        super(ViTUnet, self).__init__(down_chs, up_chs, num_class, retain_dim, out_sz, kernel_size)
        # 在 Down 之前加入 PreViT 模块
        #         self.previt = PreViT(
        #             in_channels=down_chs[0],  # 输入通道数，假设与原始输入通道一致
        #             vit_img_size=previt_img_size,
        #             vit_patch_size=previt_patch_size,
        #             vit_embed_dim=vit_embed_dim,
        #             vit_num_layers=vit_num_layers,
        #             vit_num_heads=vit_num_heads
        #         )

        # 将 bottleneck 输出（down_chs[-1]）传入 ViTModule，注意 vit_img_size 需与实际特征图尺寸匹配
        self.vit = ViTModule(
            in_channels=down_chs[-1],
            img_size=vit_img_size,
            patch_size=vit_patch_size,
            embed_dim=vit_embed_dim,
            num_layers=vit_num_layers,
            num_heads=vit_num_heads
        )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            input = self.previt(input)
            skip_connections = self.down(input)
            x = skip_connections[-1]
            # 用 ViT 模块增强全局区域注意力
            x = self.vit(x)
            x = self.bottleneck(x)
            x = self.up(x, skip_connections[::-1][1:])
            x = self.head(x)
            if self.retain_dim:
                x = F.interpolate(x, size=self.out_sz, mode='bilinear',
                                  align_corners=False)
            if self.sigmoid is not None:
                x = self.sigmoid(x)
            return x