import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    def __init__(self, in_channels=None, channel=128):
        super(FPN, self).__init__()
        if in_channels is None:
            in_channels = [64, 64, 128, 256, 512]
        self.channel_reduction_c4 = nn.Sequential(
            nn.Conv2d(in_channels[3], channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.channel_reduction_c3 = nn.Sequential(
            nn.Conv2d(in_channels[2], channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.channel_reduction_c2 = nn.Sequential(
            nn.Conv2d(in_channels[1], channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        #
        self.context_p5 = PyramidPoolingModule(in_channels[4], channel)
        self.context_p4 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.context_p3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.context_p2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channel * 4, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, c2, c3, c4, c5):
        c4 = self.channel_reduction_c4(c4)
        c3 = self.channel_reduction_c3(c3)
        c2 = self.channel_reduction_c2(c2)
        #
        p5 = self.context_p5(c5)
        p4 = self.context_p4(F.interpolate(p5, c4.size()[2:], mode='bilinear', align_corners=True) + c4)
        p3 = self.context_p3(F.interpolate(p4, c3.size()[2:], mode='bilinear', align_corners=True) + c3)
        p2 = self.context_p2(F.interpolate(p3, c2.size()[2:], mode='bilinear', align_corners=True) + c2)
        #
        p5_ = F.interpolate(p5, p2.size()[2:], mode='bilinear', align_corners=True)
        p4_ = F.interpolate(p4, p2.size()[2:], mode='bilinear', align_corners=True)
        p3_ = F.interpolate(p3, p2.size()[2:], mode='bilinear', align_corners=True)
        p_out = self.fusion_conv(torch.cat([p5_, p4_, p3_, p2], dim=1))

        return p_out


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channel=512, channel=64, pool_scales=None):
        super(PyramidPoolingModule, self).__init__()
        if pool_scales is None:
            pool_scales = [1, 2, 3, 6]
        self.pool_scale_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_scales[0]),
            nn.Conv2d(in_channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.pool_scale_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_scales[1]),
            nn.Conv2d(in_channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.pool_scale_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_scales[2]),
            nn.Conv2d(in_channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.pool_scale_4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_scales[3]),
            nn.Conv2d(in_channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channel + channel * 4, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_1 = self.pool_scale_1(x)
        x_2 = self.pool_scale_2(x)
        x_3 = self.pool_scale_3(x)
        x_4 = self.pool_scale_4(x)

        x_1 = F.interpolate(x_1, x.size()[2:], mode='bilinear', align_corners=True)
        x_2 = F.interpolate(x_2, x.size()[2:], mode='bilinear', align_corners=True)
        x_3 = F.interpolate(x_3, x.size()[2:], mode='bilinear', align_corners=True)
        x_4 = F.interpolate(x_4, x.size()[2:], mode='bilinear', align_corners=True)

        ppm_out = self.fusion_conv(torch.cat([x, x_1, x_2, x_3, x_4], dim=1))

        return ppm_out





