import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

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
    def __init__(self, enable_25D, n_classes=4, n_channels=3, bilinear=False):
        super(UNet, self).__init__()
        self.enable_25D = enable_25D
        self.fm_pools = []
        if enable_25D:
            self.SA_layers = []
            for i in range(4):
                self.SA_layers.append(nn.Conv2d(64*3*(2**i), 64*(2**i), kernel_size=1, stride=1))
                self.fm_pools.append(OrderedDict())
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
    
    def update_fm_pool(self, x, idx_slice):
        for i in range(len(self.fm_pools)):
            self.fm_pools[i][str(idx_slice)] = x[i]

    def get_fm_from_pool(self, idx_slice):
        x = []
        for i in range(len(self.fm_pools)):
            x.append(self.fm_pools[i][str(idx_slice)])
        return x

    def make_fm(self, x, idx_slice, stride, length):
        if idx_slice < stride:
            x_m = self.forward_feature(x[:, 1, :, :])
            self.update_fm_pool(x_m, idx_slice)
            return x_m, x_m, x_m
        elif idx_slice < stride * 2:
            x_m = self.forward_feature(x[:, 1, :, :])
            x_r = self.forward_feature(x[:, 2, :, :])
            x_l = self.get_fm_from_pool(idx_slice - stride)
            self.update_fm_pool(x_m, idx_slice)
            self.update_fm_pool(x_r, idx_slice+stride)
            return x_l, x_m, x_r
        elif idx_slice < length - stride:
            x_r = self.forward_feature(x[:, 2, :, :])
            x_l = self.get_fm_from_pool(idx_slice - stride)
            x_m = self.get_fm_from_pool(idx_slice)
            self.update_fm_pool(x_r, idx_slice + stride)
            return x_l, x_m, x_r
        elif idx_slice < length - 1:
            x_m = self.get_fm_from_pool(idx_slice)
            return x_m, x_m, x_m
        else:
            x_m = self.get_fm_from_pool(idx_slice)
            for i in range(len(self.fm_pools)):
                self.fm_pools[i] = OrderedDict()
            return x_m, x_m, x_m
        
    def forward_feature(self, x):
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return [x1, x2, x3, x4]

    def forward(self, x, train=False, idx_slice=0, stride=1, length=100):
        if self.enable_25D:
            x_l, x_m, x_r = self.make_fm(x, idx_slice, stride, length)
            x5 = self.down4(x_m[3])
            x1 = self.SA_layers[0](torch.cat([x_l[0], x_m[0], x_r[0]], dim=1))
            x2 = self.SA_layers[1](torch.cat([x_l[1], x_m[1], x_r[1]], dim=1))
            x3 = self.SA_layers[2](torch.cat([x_l[2], x_m[2], x_r[2]], dim=1))
            x4 = self.SA_layers[3](torch.cat([x_l[3], x_m[3], x_r[3]], dim=1))

            # skip
            x4 = x4 + x_m[3]
            x3 = x3 + x_m[2]
            x2 = x2 + x_m[1]
            x1 = x1 + x_m[0]

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

        else:
            # x = x.repeat(1, 3, 1, 1)
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

class UNet_25D(nn.Module):
    def __init__(self, enable_25D, n_classes=4, n_channels=3, bilinear=False):
        super(UNet_25D, self).__init__()
        self.enable_25D = enable_25D
        self.fm_pools = []
        if enable_25D:
            self.SA_layers =  nn.ModuleList()
            for i in range(4):
                self.SA_layers.append(nn.Conv2d(64*3*(2**i), 64*(2**i), kernel_size=1, stride=1))
                self.fm_pools.append(OrderedDict())
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
    
    def update_fm_pool(self, x, idx_slice):
        for i in range(len(self.fm_pools)):
            self.fm_pools[i][str(idx_slice)] = x[i]

    def get_fm_from_pool(self, idx_slice):
        x = []
        for i in range(len(self.fm_pools)):
            x.append(self.fm_pools[i][str(idx_slice)])
        return x

    def make_fm(self, x, idx_slice, stride, length):
        if idx_slice < stride:
            x_m = self.forward_feature(x[:, 1, :, :])
            self.update_fm_pool(x_m, idx_slice)
            return x_m, x_m, x_m
        elif idx_slice < stride * 2:
            x_m = self.forward_feature(x[:, 1, :, :])
            x_r = self.forward_feature(x[:, 2, :, :])
            x_l = self.get_fm_from_pool(idx_slice - stride)
            self.update_fm_pool(x_m, idx_slice)
            self.update_fm_pool(x_r, idx_slice+stride)
            return x_l, x_m, x_r
        elif idx_slice < length - stride:
            x_r = self.forward_feature(x[:, 2, :, :])
            x_l = self.get_fm_from_pool(idx_slice - stride)
            x_m = self.get_fm_from_pool(idx_slice)
            self.update_fm_pool(x_r, idx_slice + stride)
            return x_l, x_m, x_r
        elif idx_slice < length - 1:
            x_m = self.get_fm_from_pool(idx_slice)
            return x_m, x_m, x_m
        else:
            x_m = self.get_fm_from_pool(idx_slice)
            for i in range(len(self.fm_pools)):
                self.fm_pools[i] = OrderedDict()
            return x_m, x_m, x_m
        
    def forward_feature(self, x):
        x = x.unsqueeze(1).repeat(1, 3, 1, 1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return [x1, x2, x3, x4]

    def forward(self, x, train=False, idx_slice=0, stride=1, length=100):
        if self.enable_25D:
            x_l, x_m, x_r = self.make_fm(x, idx_slice, stride, length)
            x5 = self.down4(x_m[3])
            x1 = self.SA_layers[0](torch.cat([x_l[0], x_m[0], x_r[0]], dim=1))
            x2 = self.SA_layers[1](torch.cat([x_l[1], x_m[1], x_r[1]], dim=1))
            x3 = self.SA_layers[2](torch.cat([x_l[2], x_m[2], x_r[2]], dim=1))
            x4 = self.SA_layers[3](torch.cat([x_l[3], x_m[3], x_r[3]], dim=1))

            # skip
            x4 = x4 + x_m[3]
            x3 = x3 + x_m[2]
            x2 = x2 + x_m[1]
            x1 = x1 + x_m[0]

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

        else:
            # x = x.repeat(1, 3, 1, 1)
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

class UNet_25D_Independent(nn.Module):
    def __init__(self, enable_25D=True, n_classes=4, n_channels=3, bilinear=False):
        super(UNet_25D_Independent, self).__init__()
        self.enable_25D = enable_25D
        self.fm_pools = []
        if self.enable_25D:
            self.SA_layers = nn.ModuleList()
            for i in range(4):
                self.SA_layers.append(nn.Conv2d(64 * 3 * (2 ** i), 64 * (2 ** i), kernel_size=1, stride=1))  # conv
                self.fm_pools.append(OrderedDict())
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc_l = DoubleConv(n_channels, 64)
        self.down1_l = Down(64, 128)
        self.down2_l = Down(128, 256)
        self.down3_l = Down(256, 512)

        self.inc_m = DoubleConv(n_channels, 64)
        self.down1_m = Down(64, 128)
        self.down2_m = Down(128, 256)
        self.down3_m = Down(256, 512)

        self.inc_r = DoubleConv(n_channels, 64)
        self.down1_r = Down(64, 128)
        self.down2_r = Down(128, 256)
        self.down3_r = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x, train=True, idx_slice=-1, stride=-1, length=-1):
        if self.enable_25D:
            x_l = x[:, 0, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            x_m = x[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            x_r = x[:, 2, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
            x1_l = self.inc_l(x_l)
            x2_l = self.down1_l(x1_l)
            x3_l = self.down2_l(x2_l)
            x4_l = self.down3_l(x3_l)
            x1_m = self.inc_m(x_m)
            x2_m = self.down1_m(x1_m)
            x3_m = self.down2_m(x2_m)
            x4_m = self.down3_m(x3_m)
            x1_r = self.inc_r(x_r)
            x2_r = self.down1_r(x1_r)
            x3_r = self.down2_r(x2_r)
            x4_r = self.down3_r(x3_r)
            x5 = self.down4(x4_m)
            x1 = self.SA_layers[0](torch.cat([x1_l, x1_m, x1_r], dim=1))
            x2 = self.SA_layers[1](torch.cat([x2_l, x2_m, x2_r], dim=1))
            x3 = self.SA_layers[2](torch.cat([x3_l, x3_m, x3_r], dim=1))
            x4 = self.SA_layers[3](torch.cat([x4_l, x4_m, x4_r], dim=1))

            # skip
            x4 = x4 + x4_m
            x3 = x3 + x3_m
            x2 = x2 + x2_m
            x1 = x1 + x1_m

            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)

        else:
            # x = x.repeat(1, 3, 1, 1)
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
    channel_sum = []
    for i in feature[0]:
        channel_sum.append(torch.sum(i) * torch.sum(i))
    max_channel_index = channel_sum.index(max(channel_sum))
    max_channel = feature[0][max_channel_index]
    from torchvision import utils as vutils
    vutils.save_image(max_channel, 'features_vis/channel_max_{}.jpg'.format(name), normalize=True)


class UNet_3D(nn.Module):
    def __init__(self, in_channel=3, out_channel=4, training=True):
        super(UNet_3D, self).__init__()
        self.training = training
        self.encoder1_ = nn.Conv3d(in_channel, 64, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder1 = nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.encoder2_=   nn.Conv3d(64, 128, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder2=   nn.Conv3d(128, 128, 3, stride=1, padding=1)
        self.encoder3_=   nn.Conv3d(128, 256, 3, stride=1, padding=1)
        self.encoder3=   nn.Conv3d(256, 256, 3, stride=1, padding=1)
        self.encoder4_=   nn.Conv3d(256, 512, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv3d(512, 512, 3, stride=1, padding=1)
        self.encoder5_=   nn.Conv3d(512, 1024, 3, stride=1, padding=1)
        self.encoder5=   nn.Conv3d(1024, 1024, 3, stride=1, padding=1)
        
        self.decoder1_ = nn.Conv3d(1024, 512, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2_ =   nn.Conv3d(512, 256, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3_ =   nn.Conv3d(256, 128, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4_ =   nn.Conv3d(128, 64, 3, stride=1, padding=1)
        self.decoder5_ =   nn.Conv3d(64, 32, 3, stride=1, padding=1)
        self.decoder1 = nn.Conv3d(512, 512, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv3d(256, 256, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv3d(128, 128, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv3d(64, 64, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv3d(32, 2, 3, stride=1, padding=1)
        
        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 8, 8), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 16, 16), mode='trilinear'),
            nn.Softmax(dim =1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(512, out_channel, 1, 1),
            nn.Upsample(scale_factor=(16, 32, 32), mode='trilinear'),
            nn.Softmax(dim =1)
        )

    def forward(self, x):

        out = F.relu(F.max_pool3d(self.encoder1(self.encoder1_(x)),2,2))
        t1 = out
        out = F.relu(F.max_pool3d(self.encoder2(self.encoder2_(out)),2,2))
        t2 = out
        out = F.relu(F.max_pool3d(self.encoder3(self.encoder3_(out)),2,2))
        t3 = out
        out = F.relu(F.max_pool3d(self.encoder4(self.encoder4_(out)),2,2))
        # t4 = out
        out = F.relu(F.max_pool3d(self.encoder5(self.encoder5_(out)),2,2))
        
        # t2 = out
        out = F.relu(F.interpolate(self.decoder1(self.decoder1_(out)),scale_factor=(2,2,2),mode ='trilinear'))
        # print(out.shape,t4.shape)
        output1 = self.map1(out)
        out = F.relu(F.interpolate(self.decoder2(self.decoder2_(out)),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t3)
        output2 = self.map2(out)
        out = F.relu(F.interpolate(self.decoder3(self.decoder3_(out)),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t2)
        output3 = self.map3(out)
        out = F.relu(F.interpolate(self.decoder4(self.decoder4_(out)),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(self.decoder5_(out)),scale_factor=(2,2,2),mode ='trilinear'))
        output4 = self.map4(out)
        # print(out.shape)
        # print(output1.shape,output2.shape,output3.shape,output4.shape)
        if self.training is True:
            return output1, output2, output3, output4
        else:
            return output4

#pip install ptflops
from ptflops import get_model_complexity_info
# from torchvision.models import resnet50a
model_2D = UNet(enable_25D=False)
flops, params = get_model_complexity_info(model_2D, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
print('Flops:  ' + flops) # 41.91 GMac
print('Params: ' + params) # 31.04 M
# input("?")

model_25D = UNet_25D(enable_25D=True)
flops, params = get_model_complexity_info(model_25D, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
print('Flops:  ' + flops) # 44.387 GMac
print('Params: ' + params) # 32.08 M
# input("?")

model_25D_Independent = UNet_25D_Independent(enable_25D=True)
flops, params = get_model_complexity_info(model_25D_Independent, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
print('Flops:  ' + flops) # 64.98 GMac
print('Params: ' + params) # 41.46 M
# input("?") 

model_3D = UNet_3D()
# # input_test = torch.randn(1,1,3,224,224)
# # print(model(input_test))
# from ptflops import get_model_complexity_info
# from torchvision.models import resnet50
flops, params = get_model_complexity_info(model_3D, (3, 32, 224, 224), as_strings=True, print_per_layer_stat=False)
print('Flops:  ' + flops) # 463.08 GMac, 32_slice, overlap=0.5 ->>>>463.08/16=28.9425
print('Params: ' + params) # 84.79 M