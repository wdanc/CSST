import torch
import torch.nn as nn
import torch.nn.functional as F

from models.CSST import CSSTransformerLevel, Upsample


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

class CSSTUNet(nn.Module):
    def __init__(self, n_channels, n_classes, dims=(32,64,128,256), bilinear=True):
        super(CSSTUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, dims[0]))
        self.down1 = (Down(dims[0], dims[1]))
        self.down2 = (Down(dims[1], dims[2]))
        self.down3 = (Down(dims[2], dims[3]))
        factor = 2 if bilinear else 1
        self.down4 = (Down(dims[3], dims[3]*2 // factor))  # down(inchannel, outchannel)
        self.up1 = (Up(dims[3]*2, dims[3] // factor, bilinear))  # up(in_channel, out_channel)
        self.up2 = (Up(dims[2]*2, dims[2] // factor, bilinear))
        self.up3 = (Up(dims[1]*2, dims[1] // factor, bilinear))
        # self.up4 = (Up(dims[0]*2, dims[0], bilinear))
        self.upsample_00 = Upsample(ups_factor=4, in_chans=dims[1], up_chans=dims[1] * 16)
        self.outc = (OutConv(dims[0], n_classes))

        # CSST
        self.diff_csst5 = nn.Conv2d(dims[3] * 2, dims[3], kernel_size=3, stride=1, padding=1)
        self.diff_csst4 = nn.Conv2d(dims[3] * 2, dims[3], kernel_size=3, stride=1, padding=1)
        self.diff_csst3 = nn.Conv2d(dims[2] * 2, dims[2], kernel_size=3, stride=1, padding=1)

        self.csst5 = CSSTransformerLevel(block_size=4, num_heads=dims[3] // 32,
                                         depth=2, embed_dim=dims[3], norm_layer=nn.LayerNorm,
                                         act_layer=nn.GELU)
        self.csst4 = CSSTransformerLevel(block_size=4, num_heads=dims[3] // 32,
                                         depth=2, embed_dim=dims[3], norm_layer=nn.LayerNorm,
                                         act_layer=nn.GELU)
        self.csst3 = CSSTransformerLevel(block_size=4, num_heads=dims[2] // 32,
                                         depth=2, embed_dim=dims[2], norm_layer=nn.LayerNorm,
                                         act_layer=nn.GELU)

        self.chn_aj5 = nn.Conv2d(dims[3]*3, dims[3], kernel_size=3, padding=1)
        self.chn_aj4 = nn.Conv2d(dims[3] * 3, dims[3], kernel_size=3, padding=1)
        self.chn_aj3 = nn.Conv2d(dims[2] * 3, dims[2], kernel_size=3, padding=1)
        self.chn_aj1 = nn.Conv2d(dims[0] * 2+dims[1], dims[0], kernel_size=3, padding=1)


    def csst(self, x1, x2, diff_op, trans_op):
        diff = torch.cat((x2, x1), dim=1)
        diff = diff_op(diff) + torch.abs(x2 - x1)
        diff, media_p = trans_op(diff)
        return diff, media_p

    def forward_down(self, x):
        x1 = self.inc(x) # dim0 256 256
        x2 = self.down1(x1) # dim1 128 128
        x3 = self.down2(x2) # dim2 64 64
        x4 = self.down3(x3) # dim3 32 32
        x5 = self.down4(x4) # dim3 16 16

        return [x1, x2, x3, x4, x5]

    def forward_up(self, x5, x4, x3):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.upsample_00(x)

        return x

    def forward(self, x1, x2):
        x11, x12, x13, x14, x15 = self.forward_down(x1)
        x21, x22, x23, x24, x25 = self.forward_down(x2)

        csst_xd3, media_p3 = self.csst(x13, x23, self.diff_csst3, self.csst3)
        csst_xd4, media_p4 = self.csst(x14, x24, self.diff_csst4, self.csst4)
        csst_xd5, media_p5 = self.csst(x15, x25, self.diff_csst5, self.csst5)

        csst_xd3 = self.chn_aj3(torch.cat([csst_xd3, x13, x23], dim=1))
        csst_xd4 = self.chn_aj4(torch.cat([csst_xd4, x14, x24], dim=1))
        csst_xd5 = self.chn_aj5(torch.cat([csst_xd5, x15, x25], dim=1))

        x = self.forward_up(csst_xd5, csst_xd4, csst_xd3)
        x = self.chn_aj1(torch.cat([x, x11, x21], dim=1))
        x = self.outc(x)
        mp=[]
        mp.extend(media_p3)
        mp.extend(media_p4)
        mp.extend(media_p5)

        return x, mp