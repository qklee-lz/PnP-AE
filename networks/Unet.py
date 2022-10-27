import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, enable_25D, n_classes, n_channels=3, bilinear=False):
        super(UNet, self).__init__()
        self.enable_25D = enable_25D
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        if self.enable_25D:
            x_L = x[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            x_M = x[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            x_R = x[:, 2, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            x_L1 = self.inc(x_L)
            x_M1 = self.inc(x_M)
            x_R1 = self.inc(x_R)
            x_L2 = self.down1(x_L1)
            x_M2 = self.down1(x_M1)
            x_R2 = self.down1(x_R1)
            x_L3 = self.down2(x_L2)
            x_M3 = self.down2(x_M2)
            x_R3 = self.down2(x_R2)
            x_L4 = self.down3(x_L3)
            x_M4 = self.down3(x_M3)
            x_R4 = self.down3(x_R3)
            x5 = self.down4(x_M4)
            # print(x5.shape)
            # print(x_M4.shape)
            # print(x_M3.shape)
            # print(x_M2.shape)
            # print(x_M1.shape)
            # input("?")
            # feature_vis(x_M1, 'x_M1')
            # feature_vis(x_L1, 'x_L1')
            # feature_vis(x_R1, 'x_R1')
            
            # feature_vis(x_M4, 'x_M4')
            # feature_vis(x_L4, 'x_L4')
            # feature_vis(x_R4, 'x_R4')

            # feature_vis(x_M2, 'x_M2')
            # feature_vis(x_L2, 'x_L2')
            # feature_vis(x_R2, 'x_R2')

            # feature_vis(x_M3, 'x_M3')
            # feature_vis(x_L3, 'x_L3')
            # feature_vis(x_R3, 'x_R3')
            # input("??")

            x4 = x_L4+x_M4+x_R4
            x3 = x_L3+x_M3+x_R3
            x2 = x_L2+x_M2+x_R2
            x1 = x_L1+x_M1+x_R1
            x4 = x4+x_M4
            x3 = x3+x_M3
            x2 = x2+x_M2
            x1 = x1+x_M1
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        else:
            x = x.repeat(1,3,1,1)
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def feature_vis(feature, name):
    channel_sum=[]
    for i in feature[0]:
        channel_sum.append(torch.sum(i) * torch.sum(i))
    max_channel_index = channel_sum.index(max(channel_sum))
    max_channel = feature[0][max_channel_index]
    from  torchvision import utils as vutils
    vutils.save_image(max_channel, 'features_vis/channel_max_{}.jpg'.format(name), normalize=True)
