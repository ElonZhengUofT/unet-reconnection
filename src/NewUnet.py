import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from torchvision import transforms

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        #         self.conv = nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
        #                       padding=0, stride=1),
        #             nn.BatchNorm2d(out_channels),
        #             nn.ReLU(inplace=True),
        #             nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
        #                       padding=0, stride=1),
        #             nn.BatchNorm2d(out_channels),
        #             nn.ReLU(inplace=True)
        #         )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,padding=0, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 2* out_channels, kernel_size=kernel_size,padding=0, stride=1),
            nn.BatchNorm2d(2 * out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=kernel_size,padding=0, stride=1)
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
        return output

class Down(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super(Down, self).__init__()
        self.up_blocks = nn.ModuleList([Block(channels[i], channels[i+1],
                                              kernel_size)
                                        for i in range (len(channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skips = []
        down = x
        for block in self.up_blocks:
            down = block(down)
            skips.append(down)
            down = self.pool(down)
        return skips


class Up(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super(Up, self).__init__()
        self.channels = channels
        self.up = nn.ModuleList([nn.ConvTranspose2d(channels[i], channels[i+1],
                                                    kernel_size=2, stride=2)
                                 for i in range(len(channels)-1)])
        # 修改2：卷积部分也使用kernel_size参数，并去除padding（设为0）
        self.conv = nn.ModuleList([Block(channels[i], channels[i+1], kernel_size)
                                   for i in range(len(channels)-1)])

    def center_crop(self, skip_connection, target):
        # 修改3：用center crop方式裁剪跳跃连接，与简单版一致
        _, _, h, w = target.shape
        skip_connection = transforms.CenterCrop([h, w])(skip_connection)
        return skip_connection

    def forward(self, x: torch.Tensor,
                skip_connections: torch.Tensor) -> torch.Tensor:
        up = x
        k = 1
        len_skip = len(skip_connections)
        for i in range(len(self.channels)-1):
            up = self.up[i](up)
            skip_connection = self.center_crop(skip_connections[i], up)
            up = torch.cat([up,skip_connection], dim=1)
            up = self.conv[i](up)
            k += 1
        return up


class UNet(nn.Module):
    def __init__(self,
                 down_chs: Tuple[int, ...] = (6, 64, 128, 256),
                 up_chs: Tuple[int, ...] = (256, 128, 64),
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
         - binary_class: 是否为二分类问题（默认 True）
        """
        super(UNet, self).__init__()
        self.down = Down(down_chs, kernel_size)
        self.bottleneck = Block(down_chs[-1], down_chs[-1], kernel_size)
        self.up = Up(up_chs, kernel_size)
        self.head = nn.Conv2d(up_chs[-1], num_class, kernel_size=1)
        if num_class == 1:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None
        self.retain_dim = retain_dim
        self.out_sz = out_sz
        self.kernel_size = kernel_size
        self.binary_class = (num_class == 1)



    def forward(self, input: torch.Tensor) -> torch.Tensor:
        skip_connections = self.down(input)
        x = skip_connections[-1]
        x = self.bottleneck(x)
        x = self.up(x, skip_connections[::-1][1:])
        x = self.head(x)
        if self.retain_dim:
            x = F.interpolate(x, size=self.out_sz, mode='bilinear',
                              align_corners=False)
        if self.sigmoid is not None:
            x = self.sigmoid(x)
        return x