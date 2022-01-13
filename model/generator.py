import numpy as np
import torch
import torch.nn as nn
from base import BaseModel


# ========================================
# Generator
# ========================================
def conv_block(input_dim, output_dim, kernel_size=3, stride=1):
    seq = []
    if kernel_size != 1:
        padding = int(np.floor((kernel_size-1)/2))
        seq += [nn.ReflectionPad2d(padding)]

    seq += [nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=0, bias=True),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)]
    return nn.Sequential(*seq)


def upsample_conv_block(input_dim, output_dim, kernel_size=3):
    seq = []
    seq += [nn.UpsamplingNearest2d(scale_factor=2)]
    if kernel_size != 1:
        padding = int(np.floor((kernel_size-1)/2))
        seq += [nn.ReflectionPad2d(padding)]

    seq += [nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)]

    return nn.Sequential(*seq)


class Generator(BaseModel):
    """
    A Unet like Generator
    """
    def __init__(self, in_classes=4, out_classes=1):
        super(Generator, self).__init__()
        self.conv_block_1 = conv_block(input_dim=in_classes, output_dim=16, kernel_size=1, stride=1)
        self.conv_block_2 = conv_block(input_dim=16, output_dim=32, kernel_size=3, stride=2)
        self.conv_block_3 = conv_block(input_dim=32, output_dim=64, kernel_size=3, stride=1)
        self.conv_block_4 = conv_block(input_dim=64, output_dim=128, kernel_size=3, stride=2)
        self.conv_block_5 = conv_block(input_dim=128, output_dim=128, kernel_size=3, stride=1)
        self.conv_block_6 = conv_block(input_dim=128, output_dim=256, kernel_size=3, stride=2)
        self.conv_block_7 = conv_block(input_dim=256, output_dim=256, kernel_size=3, stride=1)
        self.conv_block_8 = conv_block(input_dim=256, output_dim=512, kernel_size=3, stride=2)
        self.conv_block_9 = conv_block(input_dim=512, output_dim=256, kernel_size=3, stride=1)

        self.up_conv_block_1 = upsample_conv_block(input_dim=256, output_dim=256, kernel_size=3)
        self.conca_conv_block_1 = conv_block(input_dim=512, output_dim=256, kernel_size=3, stride=1)
        self.up_conv_block_2 = upsample_conv_block(input_dim=256, output_dim=128, kernel_size=3)
        self.conca_conv_block_2 = conv_block(input_dim=256, output_dim=128, kernel_size=3, stride=1)
        self.up_conv_block_3 = upsample_conv_block(input_dim=128, output_dim=64, kernel_size=3)
        self.conca_conv_block_3 = conv_block(input_dim=96, output_dim=64, kernel_size=3, stride=1)
        self.up_conv_block_4 = upsample_conv_block(input_dim=64, output_dim=32, kernel_size=3)
        self.conv_block_10 = conv_block(input_dim=32, output_dim=16, kernel_size=3, stride=1)
        self.conv_block_11 = conv_block(input_dim=16, output_dim=2, kernel_size=3, stride=1)
        self.conv_block_12 = nn.Conv2d(2, out_classes, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax2d()
        self.monte_carlo_count = 5

        self.latent_conv = conv_block(input_dim=512, output_dim=256, kernel_size=1, stride=1)
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.similarity_liner = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        x = self.conv_block_1(images)
        x = self.conv_block_2(x)
        skip_1 = x

        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        skip_2 = x

        x = self.conv_block_5(x)
        x = self.conv_block_6(x)
        skip_3 = x

        x = self.conv_block_7(x)
        x = self.conv_block_8(x)
        x = self.conv_block_9(x)

        # Upsample
        x = self.up_conv_block_1(x)
        x = torch.cat([x, skip_3], 1)
        x = self.conca_conv_block_1(x)

        x = self.up_conv_block_2(x)
        x = torch.cat([x, skip_2], 1)
        x = self.conca_conv_block_2(x)

        x = self.up_conv_block_3(x)
        x = torch.cat([x, skip_1], 1)
        x = self.conca_conv_block_3(x)

        x = self.up_conv_block_4(x)
        x = self.conv_block_10(x)
        x = self.conv_block_11(x)
        x = self.conv_block_12(x)

        prob = self.sigmoid(x)
        return prob

    def inference(self, images):
        prob = self.forward(images)
        return prob


# ========================================
# LateFuseNet
# ========================================
class LateFuseNet(BaseModel):
    def __init__(self, in_classes=3, out_classes=1):
        super(LateFuseNet, self).__init__()
        self.conv_block_1 = conv_block(input_dim=in_classes, output_dim=16, kernel_size=1, stride=1)
        self.conv_block_2 = conv_block(input_dim=16, output_dim=32, kernel_size=3, stride=2)
        self.conv_block_3 = conv_block(input_dim=32, output_dim=64, kernel_size=3, stride=1)
        self.conv_block_4 = conv_block(input_dim=64, output_dim=128, kernel_size=3, stride=2)
        self.conv_block_5 = conv_block(input_dim=128, output_dim=128, kernel_size=3, stride=1)
        self.conv_block_6 = conv_block(input_dim=128, output_dim=256, kernel_size=3, stride=2)
        self.conv_block_7 = conv_block(input_dim=256, output_dim=256, kernel_size=3, stride=1)
        self.conv_block_8 = conv_block(input_dim=256, output_dim=512, kernel_size=3, stride=2)
        self.conv_block_9 = conv_block(input_dim=512, output_dim=256, kernel_size=3, stride=1)

        self.up_conv_block_1 = upsample_conv_block(input_dim=256, output_dim=256, kernel_size=3)
        self.conca_conv_block_1 = conv_block(input_dim=512, output_dim=256, kernel_size=3, stride=1)
        self.up_conv_block_2 = upsample_conv_block(input_dim=256, output_dim=128, kernel_size=3)
        self.conca_conv_block_2 = conv_block(input_dim=256, output_dim=128, kernel_size=3, stride=1)
        self.up_conv_block_3 = upsample_conv_block(input_dim=128, output_dim=64, kernel_size=3)
        self.conca_conv_block_3 = conv_block(input_dim=96, output_dim=64, kernel_size=3, stride=1)
        self.up_conv_block_4 = upsample_conv_block(input_dim=64, output_dim=32, kernel_size=3)
        self.conv_block_10 = conv_block(input_dim=32, output_dim=16, kernel_size=3, stride=1)
        self.conv_block_11 = conv_block(input_dim=16, output_dim=2, kernel_size=3, stride=1)
        self.conv_block_12 = nn.Conv2d(2, out_classes, kernel_size=3, stride=1, padding=1)
        self.conv_fuse = nn.Conv2d(2, out_classes, kernel_size=1, stride=1, padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, images, partials):
        x = self.conv_block_1(images)
        x = self.conv_block_2(x)
        skip_1 = x

        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        skip_2 = x

        x = self.conv_block_5(x)
        x = self.conv_block_6(x)
        skip_3 = x

        x = self.conv_block_7(x)
        x = self.conv_block_8(x)
        x = self.conv_block_9(x)

        # Upsample
        x = self.up_conv_block_1(x)
        x = torch.cat([x, skip_3], 1)
        x = self.conca_conv_block_1(x)

        x = self.up_conv_block_2(x)
        x = torch.cat([x, skip_2], 1)
        x = self.conca_conv_block_2(x)

        x = self.up_conv_block_3(x)
        x = torch.cat([x, skip_1], 1)
        x = self.conca_conv_block_3(x)

        x = self.up_conv_block_4(x)
        x = self.conv_block_10(x)
        x = self.conv_block_11(x)
        x = self.conv_block_12(x)

        # fuse
        x = torch.cat([x, partials], 1)
        x = self.conv_fuse(x)

        prob = self.sigmoid(x)
        return prob


# ========================================
# FuseSat
# - Satellite Branch for FuseNet
# ========================================
class FuseSat(BaseModel):
    """
    A Unet like FuseSat
    """
    def __init__(self, in_classes=3, out_classes=1, use_sigmoid=True):
        super(FuseSat, self).__init__()
        self.conv_block_1 = conv_block(input_dim=in_classes, output_dim=16, kernel_size=1, stride=1)
        self.conv_block_2 = conv_block(input_dim=16, output_dim=32, kernel_size=3, stride=2)
        self.conv_block_3 = conv_block(input_dim=32, output_dim=64, kernel_size=3, stride=1)
        self.conv_block_4 = conv_block(input_dim=64, output_dim=128, kernel_size=3, stride=2)
        self.conv_block_5 = conv_block(input_dim=128, output_dim=128, kernel_size=3, stride=1)
        self.conv_block_6 = conv_block(input_dim=128, output_dim=256, kernel_size=3, stride=2)
        self.conv_block_7 = conv_block(input_dim=256, output_dim=256, kernel_size=3, stride=1)
        self.conv_block_8 = conv_block(input_dim=256, output_dim=512, kernel_size=3, stride=2)
        self.conv_block_9 = conv_block(input_dim=512, output_dim=256, kernel_size=3, stride=1)

        self.up_conv_block_1 = upsample_conv_block(input_dim=256, output_dim=256, kernel_size=3)
        self.conca_conv_block_1 = conv_block(input_dim=512, output_dim=256, kernel_size=3, stride=1)
        self.up_conv_block_2 = upsample_conv_block(input_dim=256, output_dim=128, kernel_size=3)
        self.conca_conv_block_2 = conv_block(input_dim=256, output_dim=128, kernel_size=3, stride=1)
        self.up_conv_block_3 = upsample_conv_block(input_dim=128, output_dim=64, kernel_size=3)
        self.conca_conv_block_3 = conv_block(input_dim=96, output_dim=64, kernel_size=3, stride=1)
        self.up_conv_block_4 = upsample_conv_block(input_dim=64, output_dim=32, kernel_size=3)
        self.conv_block_10 = conv_block(input_dim=32, output_dim=16, kernel_size=3, stride=1)
        self.conv_block_11 = conv_block(input_dim=16, output_dim=2, kernel_size=3, stride=1)
        self.conv_block_12 = nn.Conv2d(2, out_classes, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax2d()
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, sats):
        # Encoder
        x = self.conv_block_1(sats)
        x = self.conv_block_2(x)
        skip_1 = x

        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        skip_2 = x

        x = self.conv_block_5(x)
        x = self.conv_block_6(x)
        skip_3 = x

        x = self.conv_block_7(x)
        x = self.conv_block_8(x)
        x = self.conv_block_9(x)
        fm1 = x

        # Decoder
        x = self.up_conv_block_1(x)
        x = torch.cat([x, skip_3], 1)
        x = self.conca_conv_block_1(x)
        fm2 = x

        x = self.up_conv_block_2(x)
        x = torch.cat([x, skip_2], 1)
        x = self.conca_conv_block_2(x)
        fm3 = x

        x = self.up_conv_block_3(x)
        x = torch.cat([x, skip_1], 1)
        x = self.conca_conv_block_3(x)
        fm4 = x

        x = self.up_conv_block_4(x)
        x = self.conv_block_10(x)
        x = self.conv_block_11(x)
        fm5 = x

        x = self.conv_block_12(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x, [fm1, fm2, fm3, fm4, fm5]


# ========================================
# FusePar
# - Partial Branch for FuseNet
# ========================================
class FusePar(BaseModel):
    """
    A Unet like FusePar
    """
    def __init__(self, in_classes=1, out_classes=1, use_sigmoid=True):
        super(FusePar, self).__init__()
        self.conv_block_1 = conv_block(input_dim=in_classes, output_dim=16, kernel_size=1, stride=1)
        self.conv_block_2 = conv_block(input_dim=16, output_dim=32, kernel_size=3, stride=2)
        self.conv_block_3 = conv_block(input_dim=32, output_dim=64, kernel_size=3, stride=1)
        self.conv_block_4 = conv_block(input_dim=64, output_dim=128, kernel_size=3, stride=2)
        self.conv_block_5 = conv_block(input_dim=128, output_dim=128, kernel_size=3, stride=1)
        self.conv_block_6 = conv_block(input_dim=128, output_dim=256, kernel_size=3, stride=2)
        self.conv_block_7 = conv_block(input_dim=256, output_dim=256, kernel_size=3, stride=1)
        self.conv_block_8 = conv_block(input_dim=256, output_dim=512, kernel_size=3, stride=2)
        self.conv_block_9 = conv_block(input_dim=512, output_dim=256, kernel_size=3, stride=1)

        self.up_conv_block_1 = upsample_conv_block(input_dim=512, output_dim=256, kernel_size=3)
        self.conca_conv_block_1 = conv_block(input_dim=512, output_dim=256, kernel_size=3, stride=1)
        self.up_conv_block_2 = upsample_conv_block(input_dim=512, output_dim=128, kernel_size=3)
        self.conca_conv_block_2 = conv_block(input_dim=256, output_dim=128, kernel_size=3, stride=1)
        self.up_conv_block_3 = upsample_conv_block(input_dim=256, output_dim=64, kernel_size=3)
        self.conca_conv_block_3 = conv_block(input_dim=96, output_dim=64, kernel_size=3, stride=1)
        self.up_conv_block_4 = upsample_conv_block(input_dim=128, output_dim=32, kernel_size=3)
        self.conv_block_10 = conv_block(input_dim=32, output_dim=16, kernel_size=3, stride=1)
        self.conv_block_11 = conv_block(input_dim=16, output_dim=2, kernel_size=3, stride=1)
        self.conv_block_12 = nn.Conv2d(4, out_classes, kernel_size=3, stride=1, padding=1)

        self.adaptor_0 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.adaptor_1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.adaptor_2 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.adaptor_3 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.adaptor_4 = nn.Conv2d(2, 2, kernel_size=1, stride=1, padding=0)

        self.softmax = nn.Softmax2d()
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, partials, fms):
        assert len(fms) == 5, "The length of feature maps set should be 5"
        # Encoder
        x = self.conv_block_1(partials)
        x = self.conv_block_2(x)
        skip_1 = x

        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        skip_2 = x

        x = self.conv_block_5(x)
        x = self.conv_block_6(x)
        skip_3 = x

        x = self.conv_block_7(x)
        x = self.conv_block_8(x)
        x = self.conv_block_9(x)
        fm1 = x

        # Decoder
        fms_0 = self.adaptor_0(fms[0])
        x = torch.cat([x, fms_0], 1)
        x = self.up_conv_block_1(x)
        x = torch.cat([x, skip_3], 1)
        x = self.conca_conv_block_1(x)
        fm2 = x

        fms_1 = self.adaptor_1(fms[1])
        x = torch.cat([x, fms_1], 1)
        x = self.up_conv_block_2(x)
        x = torch.cat([x, skip_2], 1)
        x = self.conca_conv_block_2(x)
        fm3 = x

        fms_2 = self.adaptor_2(fms[2])
        x = torch.cat([x, fms_2], 1)
        x = self.up_conv_block_3(x)
        x = torch.cat([x, skip_1], 1)
        x = self.conca_conv_block_3(x)
        fm4 = x

        fms_3 = self.adaptor_3(fms[3])
        x = torch.cat([x, fms_3], 1)
        x = self.up_conv_block_4(x)
        x = self.conv_block_10(x)
        x = self.conv_block_11(x)
        fm5 = x

        fms_4 = self.adaptor_4(fms[4])
        x = torch.cat([x, fms_4], 1)
        x = self.conv_block_12(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x, [fm1, fm2, fm3, fm4, fm5]


# ========================================
# GFM
# - Gated Fusion Module for FuseNet
# ========================================
class GFM(BaseModel):
    """
    Gated Fusion Module
    """
    def __init__(self, in_classes=256, out_classes=2, is_first=False):
        super(GFM, self).__init__()
        self.is_first = is_first
        self.adaptor_s = nn.Conv2d(in_classes, in_classes, kernel_size=1, stride=1, padding=0)
        self.adaptor_p = nn.Conv2d(in_classes, in_classes, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv_block_1 = conv_block(input_dim=in_classes * 2, output_dim=in_classes, kernel_size=3, stride=1)
        self.conv_block_2 = conv_block(input_dim=in_classes, output_dim=in_classes, kernel_size=3, stride=1)
        self.conv_block_3 = conv_block(input_dim=in_classes, output_dim=out_classes, kernel_size=1, stride=1)
        self.softmax = nn.Softmax2d()

    def forward(self, g_prev, f_s, f_p):
        # Upsample G_{i-1} if it is not the 1st GFM
        if not self.is_first:
            g_prev = self.upsample(g_prev)

        # Adaptor
        a_s = self.adaptor_s(f_s)
        a_p = self.adaptor_p(f_p)

        # Selector
        a = torch.cat([a_s, a_p], dim=1)
        x = self.conv_block_1(a)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        if not self.is_first:
            x = x + g_prev
        g_next = x
        x = self.softmax(x)

        g_s = x[:, 0, :, :]
        g_s = g_s.unsqueeze(1)
        g_p = x[:, 1, :, :]
        g_p = g_p.unsqueeze(1)
        a_s = a_s * g_s
        a_p = a_p * g_p
        a_f = a_s + a_p
        return g_next, a_f


# ========================================
# FuseMer
# - Merge Branch for FuseNet
# ========================================
class FuseMer(BaseModel):
    """
    A Unet like FuseMer
    """
    def __init__(self, out_classes=1, use_gfm=False, use_sigmoid=True):
        super(FuseMer, self).__init__()
        self.use_gfm = use_gfm
        if use_gfm:
            self.gfm_1 = GFM(in_classes=256, is_first=True)
            self.gfm_2 = GFM(in_classes=256)
            self.gfm_3 = GFM(in_classes=128)
            self.gfm_4 = GFM(in_classes=64)
            self.gfm_5 = GFM(in_classes=2)
        else:
            self.gfm_1 = nn.Conv2d(256 * 2, 256, kernel_size=1, stride=1, padding=0)
            self.gfm_2 = nn.Conv2d(256 * 2, 256, kernel_size=1, stride=1, padding=0)
            self.gfm_3 = nn.Conv2d(128 * 2, 128, kernel_size=1, stride=1, padding=0)
            self.gfm_4 = nn.Conv2d(64 * 2, 64, kernel_size=1, stride=1, padding=0)
            self.gfm_5 = nn.Conv2d(2 * 2, 2, kernel_size=1, stride=1, padding=0)

        self.up_conv_block_1 = upsample_conv_block(input_dim=256, output_dim=256, kernel_size=3)
        self.conca_conv_block_1 = conv_block(input_dim=512, output_dim=256, kernel_size=3, stride=1)
        self.up_conv_block_2 = upsample_conv_block(input_dim=256, output_dim=128, kernel_size=3)
        self.conca_conv_block_2 = conv_block(input_dim=256, output_dim=128, kernel_size=3, stride=1)
        self.up_conv_block_3 = upsample_conv_block(input_dim=128, output_dim=64, kernel_size=3)
        self.conca_conv_block_3 = conv_block(input_dim=128, output_dim=64, kernel_size=3, stride=1)
        self.up_conv_block_4 = upsample_conv_block(input_dim=64, output_dim=32, kernel_size=3)
        self.conv_block_1 = conv_block(input_dim=34, output_dim=16, kernel_size=3, stride=1)
        self.conv_block_2 = conv_block(input_dim=16, output_dim=2, kernel_size=3, stride=1)
        self.conv_block_3 = nn.Conv2d(2, out_classes, kernel_size=3, stride=1, padding=1)

        self.softmax = nn.Softmax2d()
        self.use_sigmoid = use_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_s, f_p):
        # GFMs
        if self.use_gfm:
            g_0 = None
            g_1, a_f_1 = self.gfm_1(g_0, f_s[0], f_p[0])
            g_2, a_f_2 = self.gfm_2(g_1, f_s[1], f_p[1])
            g_3, a_f_3 = self.gfm_3(g_2, f_s[2], f_p[2])
            g_4, a_f_4 = self.gfm_4(g_3, f_s[3], f_p[3])
            g_5, a_f_5 = self.gfm_5(g_4, f_s[4], f_p[4])

            # Decoder
            x = a_f_1
            x = self.up_conv_block_1(x)
            x = torch.cat([x, a_f_2], dim=1)
            x = self.conca_conv_block_1(x)
            x = x + a_f_2

            x = self.up_conv_block_2(x)
            x = torch.cat([x, a_f_3], dim=1)
            x = self.conca_conv_block_2(x)
            x = x + a_f_3

            x = self.up_conv_block_3(x)
            x = torch.cat([x, a_f_4], dim=1)
            x = self.conca_conv_block_3(x)
            x = x + a_f_4

            x = self.up_conv_block_4(x)
            x = torch.cat([x, a_f_5], dim=1)
            x = self.conv_block_1(x)
            x = self.conv_block_2(x)
            x = x + a_f_5

            x = self.conv_block_3(x)
        else:
            a_f_1 = self.gfm_1(torch.cat([f_s[0], f_p[0]], dim=1))
            a_f_2 = self.gfm_2(torch.cat([f_s[1], f_p[1]], dim=1))
            a_f_3 = self.gfm_3(torch.cat([f_s[2], f_p[2]], dim=1))
            a_f_4 = self.gfm_4(torch.cat([f_s[3], f_p[3]], dim=1))
            a_f_5 = self.gfm_5(torch.cat([f_s[4], f_p[4]], dim=1))

            # Decoder
            x = a_f_1
            x = self.up_conv_block_1(x)
            x = torch.cat([x, a_f_2], dim=1)
            x = self.conca_conv_block_1(x)
            x = x + a_f_2

            x = self.up_conv_block_2(x)
            x = torch.cat([x, a_f_3], dim=1)
            x = self.conca_conv_block_2(x)
            x = x + a_f_3

            x = self.up_conv_block_3(x)
            x = torch.cat([x, a_f_4], dim=1)
            x = self.conca_conv_block_3(x)
            x = x + a_f_4

            x = self.up_conv_block_4(x)
            x = torch.cat([x, a_f_5], dim=1)
            x = self.conv_block_1(x)
            x = self.conv_block_2(x)
            x = x + a_f_5

            x = self.conv_block_3(x)

        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x


# ========================================
# McGenerator
# - Generator of McGAN
# ========================================
def mc_conv_block(input_dim, output_dim, kernel_size=3, stride=1, is_first=False, is_last=False, is_dis=False):
    seq = []
    if kernel_size != 1:
        padding = int(np.floor((kernel_size-1)/2))
        seq += [nn.ReflectionPad2d(padding)]

    if not is_dis:
        seq += [nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=0, bias=True),
                nn.InstanceNorm2d(output_dim)]
        if not is_first and not is_last:
            seq += [nn.ReLU(inplace=True)]
    else:
        seq += [nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=0, bias=True)]
        if not is_first and not is_last:
            seq += [nn.InstanceNorm2d(output_dim)]
        if not is_last:
            seq += [nn.LeakyReLU(inplace=True)]
    return nn.Sequential(*seq)


def mc_deconv_block(input_dim, output_dim, kernel_size=3, stride=2):
    seq = []
    padding = int(np.floor((kernel_size-1)/2))

    seq += [nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=padding, bias=True),
            nn.InstanceNorm2d(output_dim),
            nn.ReLU(inplace=True)]
    return nn.Sequential(*seq)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class McGenerator(BaseModel):
    """
    Generator of McGAN
    """
    def __init__(self, in_classes=4, out_classes=1):
        super(McGenerator, self).__init__()
        self.block1 = mc_conv_block(input_dim=in_classes, output_dim=64, kernel_size=7, stride=1, is_first=True)
        self.block2 = mc_conv_block(input_dim=64, output_dim=128, kernel_size=3, stride=2)
        self.block3 = mc_conv_block(input_dim=128, output_dim=256, kernel_size=3, stride=2)
        self.block4 = ResidualBlock(in_channels=256, out_channels=256)
        self.block5 = ResidualBlock(in_channels=256, out_channels=256)
        self.block6 = ResidualBlock(in_channels=256, out_channels=256)
        self.block7 = ResidualBlock(in_channels=256, out_channels=256)
        self.block8 = mc_deconv_block(input_dim=256, output_dim=256, kernel_size=3, stride=2)
        self.block9 = mc_deconv_block(input_dim=256, output_dim=128, kernel_size=3, stride=2)
        self.block10 = mc_conv_block(input_dim=128, output_dim=64, kernel_size=7, stride=1)
        self.block11 = mc_conv_block(input_dim=64, output_dim=128, kernel_size=3, stride=2)
        self.block12 = mc_conv_block(input_dim=128, output_dim=256, kernel_size=3, stride=2)
        self.block13 = ResidualBlock(in_channels=256, out_channels=256)
        self.block14 = ResidualBlock(in_channels=256, out_channels=256)
        self.block15 = ResidualBlock(in_channels=256, out_channels=256)
        self.block16 = ResidualBlock(in_channels=256, out_channels=256)
        self.block17 = mc_deconv_block(input_dim=256, output_dim=256, kernel_size=3, stride=2)
        self.block18 = mc_deconv_block(input_dim=256, output_dim=128, kernel_size=3, stride=2)
        self.block19 = mc_conv_block(input_dim=128, output_dim=out_classes, kernel_size=7, stride=1, is_last=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        skip1 = x

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = x + skip1

        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        skip2 = x

        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = x + skip2

        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)

        x = self.sigmoid(x)
        return x


# ========================================
# McDiscriminator
# - Discriminator of McGAN
# ========================================
class McDiscriminator(BaseModel):
    """
    Discriminator of McGAN
    """
    def __init__(self, in_classes=4, out_classes=1):
        super(McDiscriminator, self).__init__()
        self.block1 = mc_conv_block(input_dim=in_classes, output_dim=64, kernel_size=4, stride=2, is_first=True, is_dis=True)
        self.block2 = mc_conv_block(input_dim=64, output_dim=128, kernel_size=4, stride=2, is_dis=True)
        self.block3 = mc_conv_block(input_dim=128, output_dim=256, kernel_size=4, stride=2, is_dis=True)
        self.block4 = mc_conv_block(input_dim=256, output_dim=512, kernel_size=4, stride=1, is_dis=True)
        self.block5 = mc_conv_block(input_dim=512, output_dim=out_classes, kernel_size=4, stride=1, is_last=True, is_dis=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x.view(x.size(0), -1)
