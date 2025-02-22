import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        # 修改1：将卷积层的padding从1改为0，使得每个3×3卷积减少2个像素，与之前UNet一致
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.conv(x)
        down = self.pool(conv_out)
        skip_connection = conv_out  # 保存未池化的输出作为跳跃连接
        return down, skip_connection

class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # 修改1：这里同样将卷积层的padding从1改为0
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def center_crop(self, tensor, target_height, target_width):
        # 修改2：用center crop方式裁剪跳跃连接，与原UNet的CenterCrop一致
        _, _, h, w = tensor.size()
        start_y = (h - target_height) // 2
        start_x = (w - target_width) // 2
        return tensor[:, :, start_y:start_y+target_height, start_x:start_x+target_width]

    def forward(self, x: torch.Tensor, skip_connection: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 修改2：使用center crop裁剪skip_connection，使其与x的尺寸匹配
        skip_connection = self.center_crop(skip_connection, x.size(2), x.size(3))
        x = torch.cat([skip_connection, x], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, retain_dim: bool = False, out_sz: Tuple[int, int] = (572,572), disable_skip_connections: bool = False):
        """
        修改后的UNet，与之前UNet在输入输出格式和尺寸上保持一致：
         - 输入：形状为 (B, 3, H, W) 的RGB图像
         - 默认输出：经过一系列无padding的3×3卷积，输出尺寸为 (B, 1, H-16, W-16)；如果设置retain_dim=True，则通过插值调整为out_sz（默认572×572）
        """
        super(UNet, self).__init__()
        self.disable_skip_connections = disable_skip_connections
        self.retain_dim = retain_dim
        self.out_sz = out_sz

        # 修改3：调整输入输出通道数，与之前UNet一致（Encoder通道：3->64->128）
        self.down1 = Down(in_channels, 64)
        self.down2 = Down(64, 128)

        # 修改4：Bottleneck中的卷积同样去掉padding（padding=0），保证尺寸减少
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 修改5：调整Up模块的通道数
        # 原始UNet的decoder：dec_chs=(128, 64)，所以第一个Up模块应从64通道（bottleneck）结合跳跃连接（128）输出64通道
        self.up2 = Up(64, 128, 64)
        # 第二个Up模块：输入64通道，跳跃连接64通道，输出64通道
        self.up1 = Up(64, 64, 64)

        # 修改6：最终1×1卷积，将通道数从64映射到1（单通道分割输出）
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)
        # 如果是二分类（out_channels==1），添加sigmoid激活
        if out_channels == 1:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 保存原始输入尺寸，便于后续插值调整（如果需要）
        input_shape = x.shape[2:]

        down1, skip1 = self.down1(x)   # down1尺寸： (B,64, H-4, W-4)；skip1尺寸同下
        down2, skip2 = self.down2(down1) # down2尺寸： (B,128, ? , ?)

        bottleneck = self.bottleneck(down2)

        up2 = self.up2(bottleneck, skip2 if not self.disable_skip_connections else torch.zeros_like(skip2))
        up1 = self.up1(up2, skip1 if not self.disable_skip_connections else torch.zeros_like(skip1))

        output = self.head(up1)

        # 修改7：如果设置retain_dim为True，则对输出进行插值调整到指定尺寸
        if self.retain_dim:
            output = F.interpolate(output, size=self.out_sz, mode='bilinear', align_corners=False)
        if self.sigmoid is not None:
            output = self.sigmoid(output)
        return output
