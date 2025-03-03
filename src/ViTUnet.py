from src.NewUnet import UNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models.vision_transformer import Encoder as ViTBlock
class ViTUnet(UNet):
    def __init__(self,
                 down_chs: Tuple[int, ...] = (6, 64, 128, 256),
                 up_chs: Tuple[int, ...] = (256, 128, 64),
                 num_class: int = 1,
                 retain_dim: bool = False,
                 out_sz: Tuple[int, int] = (572, 572),
                 kernel_size: int = 3,
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
        super(ViTUnet, self).__init__(down_chs, up_chs, num_class, retain_dim, out_sz, kernel_size)
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