import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size)
    
    def forward(self, x):
        print(x.shape)
        return self.conv2(self.relu(self.conv1(x)))

class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024), kernel_size=3):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1], kernel_size) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            print(f"Encoder: {x.shape}")
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), kernel_size=3):
        super().__init__()
        self.chs        = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1], kernel_size) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        k = 0
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            print(
                f'{k} th round, x.shape: {x.shape}, enc_ftrs.shape: {enc_ftrs.shape}')
            x        = torch.cat([x, enc_ftrs], dim=1)
            print(f'{k} th round, x.shape: {x.shape}')
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs

class UNet(nn.Module):
    def __init__(
            self,
            enc_chs=(3,64,128),
            dec_chs=(128, 64),
            num_class=1,
            retain_dim=False,
            out_sz=(572,572),
            kernel_size=3
        ):
        """
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/abs/1505.04597
        https://amaarora.github.io/2020/09/13/unet.html
        """
        super().__init__()
        self.encoder     = Encoder(enc_chs, kernel_size)
        self.decoder     = Decoder(dec_chs, kernel_size)
        self.head        = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.sigmoid     = nn.Sigmoid()
        self.retain_dim  = retain_dim
        self.out_sz      = out_sz
        self.binary      = (num_class == 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        out_sz   = self.out_sz
        if self.retain_dim:
            out = F.interpolate(out, out_sz)
        if self.binary:
            out = self.sigmoid(out)   
        return out
