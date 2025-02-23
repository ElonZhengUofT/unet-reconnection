import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(Down, self).__init__()
        # 修改1：使用kernel_size参数，并去除padding（设为0），使每个3×3卷积减少尺寸，与简单版一致
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=0, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      padding=0, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv(x)
        down = self.pool(conv_out)
        # 返回池化后的结果以及未经池化的输出作为跳跃连接
        return down, conv_out


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 kernel_size: int):
        super(Up, self).__init__()
        # 修改2：使用kernel_size参数；此处的转置卷积不受padding影响
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2,
                                     stride=2)
        # 修改2：卷积部分也使用kernel_size参数，并去除padding（设为0）
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels,
                      kernel_size=kernel_size, padding=0, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                      padding=0, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def center_crop(self, tensor, target_height, target_width):
        # 修改3：用center crop方式裁剪跳跃连接，与简单版一致
        _, _, h, w = tensor.size()
        start_y = (h - target_height) // 2
        start_x = (w - target_width) // 2
        return tensor[:, :, start_y:start_y + target_height,
               start_x:start_x + target_width]

    def forward(self, x: torch.Tensor,
                skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip_connection = self.center_crop(skip_connection, x.size(2),
                                           x.size(3))
        x = torch.cat([skip_connection, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self,
                 enc_chs: Tuple[int, ...] = (3, 64, 128),
                 dec_chs: Tuple[int, ...] = (128, 64),
                 num_class: int = 1,
                 retain_dim: bool = False,
                 out_sz: Tuple[int, int] = (572, 572),
                 kernel_size: int = 3):
        """
        修改后的复杂版 UNet，其输入参数与简单版完全一致：
         - enc_chs: Encoder 各层的通道数（默认 (3, 64, 128)）
         - dec_chs: Decoder 各层的输出通道数（默认 (128, 64)）
         - num_class: 输出类别数（默认 1）
         - retain_dim: 是否通过插值调整输出尺寸到 out_sz（默认 False）
         - out_sz: 当 retain_dim 为 True 时的输出尺寸（默认 (572,572)）
         - kernel_size: 卷积核大小（默认 3）
        """
        super(UNet, self).__init__()
        # 修改4：根据 enc_chs 构造 Down 模块
        self.down1 = Down(enc_chs[0], enc_chs[1], kernel_size)
        self.down2 = Down(enc_chs[1], enc_chs[2], kernel_size)

        # 修改5：Bottleneck 模块将 Encoder 最后输出的通道数转换为 Decoder 第一层所需的通道数
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_chs[-1], dec_chs[0], kernel_size=kernel_size,
                      padding=0, stride=1),
            nn.BatchNorm2d(dec_chs[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_chs[0], dec_chs[0], kernel_size=kernel_size,
                      padding=0, stride=1),
            nn.BatchNorm2d(dec_chs[0]),
            nn.ReLU(inplace=True)
        )

        # 修改6：根据 dec_chs 和对应跳跃连接的通道数构造 Up 模块
        # 第一 Up 模块：输入 bottleneck 输出 dec_chs[0]，结合来自 down2 的跳跃连接（通道数 enc_chs[2]）
        self.up2 = Up(dec_chs[0], enc_chs[2], dec_chs[0], kernel_size)
        # 第二 Up 模块：输入上一步输出 dec_chs[0]，结合来自 down1 的跳跃连接（通道数 enc_chs[1]），输出 dec_chs[1]
        self.up1 = Up(dec_chs[0], enc_chs[1], dec_chs[1], kernel_size)

        # 修改7：Head 层将最后的 Decoder 输出映射到 num_class 通道
        self.head = nn.Conv2d(dec_chs[1], num_class, kernel_size=1)
        if num_class == 1:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None

        self.retain_dim = retain_dim
        self.out_sz = out_sz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[2:]
        down1, skip1 = self.down1(x)  # skip1 尺寸为 (B, enc_chs[1], H-?, W-?)
        down2, skip2 = self.down2(down1)  # skip2 尺寸为 (B, enc_chs[2], ...)

        bottleneck = self.bottleneck(down2)

        up2 = self.up2(bottleneck, skip2)
        up1 = self.up1(up2, skip1)

        output = self.head(up1)

        # 修改8：如果 retain_dim 为 True，则用插值将输出调整到指定尺寸 out_sz
        if self.retain_dim:
            output = F.interpolate(output, size=self.out_sz, mode='bilinear',
                                   align_corners=False)
        if self.sigmoid is not None:
            output = self.sigmoid(output)
        return output
