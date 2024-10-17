import torch.nn as nn
import torch

from models.CSST import CSSTransformerLevel, Upsample

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output


class up(nn.Module):
    def __init__(self, in_ch, bilinear=False):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

    def forward(self, x):

        x = self.up(x)
        return x


class CSSTUNetPP(nn.Module):
    def __init__(self, n_channels=3, n_classes=2):
        super(CSSTUNetPP, self).__init__()
        torch.nn.Module.dump_patches = True
        n1 = 32     # the initial number of channels of feature map
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv0_0 = conv_block_nested(n_channels, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        # self.Up1_0 = up(filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        # self.Up2_0 = up(filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.Up3_0 = up(filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])
        self.Up4_0 = up(filters[4])

        # self.conv0_1 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        # self.conv1_1 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        # self.Up1_1 = up(filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        # self.Up2_1 = up(filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])
        self.Up3_1 = up(filters[3])

        # self.conv0_2 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        # self.conv1_2 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])
        # self.Up1_2 = up(filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])
        # self.Up2_2 = up(filters[2])

        # self.conv0_3 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])
        # self.conv1_3 = conv_block_nested(filters[1] * 4 + filters[2], filters[1], filters[1])
        # self.Up1_3 = up(filters[1])

        # self.conv0_4 = conv_block_nested(filters[0] * 5 + filters[1], filters[0], filters[0])

        # self.final1 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        # self.final2 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        # self.final3 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        # self.final4 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.conv_final = nn.Conv2d(filters[0], n_classes, kernel_size=1)

        self.upsample_00 = Upsample(ups_factor=4, in_chans=filters[2], up_chans=filters[2] * 16)

        # CSST
        self.diff_csst4 = nn.Conv2d(filters[4] * 2, filters[4], kernel_size=3, stride=1, padding=1)
        self.diff_csst3 = nn.Conv2d(filters[3] * 2, filters[3], kernel_size=3, stride=1, padding=1)
        self.diff_csst2 = nn.Conv2d(filters[2] * 2, filters[2], kernel_size=3, stride=1, padding=1)

        self.csst4 = CSSTransformerLevel(block_size=4, num_heads=filters[4] // 32,
                                         depth=2, embed_dim=filters[4], norm_layer=nn.LayerNorm,
                                         act_layer=nn.GELU)
        self.csst3 = CSSTransformerLevel(block_size=4, num_heads=filters[3] // 32,
                                         depth=2, embed_dim=filters[3], norm_layer=nn.LayerNorm,
                                         act_layer=nn.GELU)
        self.csst2 = CSSTransformerLevel(block_size=4, num_heads=filters[2] // 32,
                                         depth=2, embed_dim=filters[2], norm_layer=nn.LayerNorm,
                                         act_layer=nn.GELU)

        self.chn_aj4 = nn.Conv2d(filters[4] * 3, filters[4], kernel_size=3, padding=1)
        self.chn_aj3 = nn.Conv2d(filters[3] * 3, filters[3], kernel_size=3, padding=1)
        self.chn_aj2 = nn.Conv2d(filters[2] * 3, filters[2], kernel_size=3, padding=1)
        self.chn_aj1 = nn.Conv2d(filters[0] * 2 + filters[2], filters[0], kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def csst(self, x1, x2, diff_op, trans_op):
        diff = torch.cat((x2, x1), dim=1)
        diff = diff_op(diff) + torch.abs(x2 - x1)
        diff, media_p = trans_op(diff)
        return diff, media_p


    def forward(self, xA, xB):
        '''xA'''
        x0_0A = self.conv0_0(xA)
        x1_0A = self.conv1_0(self.pool(x0_0A))
        x2_0A = self.conv2_0(self.pool(x1_0A))
        x3_0A = self.conv3_0(self.pool(x2_0A))
        x4_0A = self.conv4_0(self.pool(x3_0A))
        '''xB'''
        x0_0B = self.conv0_0(xB)
        x1_0B = self.conv1_0(self.pool(x0_0B))
        x2_0B = self.conv2_0(self.pool(x1_0B))
        x3_0B = self.conv3_0(self.pool(x2_0B))
        x4_0B = self.conv4_0(self.pool(x3_0B))

        csst_xd2, media_p2 = self.csst(x2_0A, x2_0B, self.diff_csst2, self.csst2)
        csst_xd3, media_p3 = self.csst(x3_0A, x3_0B, self.diff_csst3, self.csst3)
        csst_xd4, media_p4 = self.csst(x4_0A, x4_0B, self.diff_csst4, self.csst4)

        csst_xd2 = self.chn_aj2(torch.cat([csst_xd2, x2_0A, x2_0B], dim=1))
        csst_xd3 = self.chn_aj3(torch.cat([csst_xd3, x3_0A, x3_0B], dim=1))
        csst_xd4 = self.chn_aj4(torch.cat([csst_xd4, x4_0A, x4_0B], dim=1))

        # x0_1 = self.conv0_1(torch.cat([x0_0A, x0_0B, self.Up1_0(x1_0B)], 1))
        # x1_1 = self.conv1_1(torch.cat([x1_0A, x1_0B, self.Up2_0(csst_xd2)], 1))
        # x0_2 = self.conv0_2(torch.cat([x0_0A, x0_0B, x0_1, self.Up1_1(x1_1)], 1))


        x2_1 = self.conv2_1(torch.cat([csst_xd2, self.Up3_0(csst_xd3)], 1))
        # x1_2 = self.conv1_2(torch.cat([x1_0A, x1_0B, x1_1, self.Up2_1(x2_1)], 1))
        # x0_3 = self.conv0_3(torch.cat([x0_0A, x0_0B, x0_1, x0_2, self.Up1_2(x1_2)], 1))

        x3_1 = self.conv3_1(torch.cat([csst_xd3, self.Up4_0(csst_xd4)], 1))
        x2_2 = self.conv2_2(torch.cat([csst_xd2, x2_1, self.Up3_1(x3_1)], 1))
        # x1_3 = self.conv1_3(torch.cat([x1_0A, x1_0B, x1_1, x1_2, self.Up2_2(x2_2)], 1))
        # x0_4 = self.conv0_4(torch.cat([x0_0A, x0_0B, x0_1, x0_2, x0_3, self.Up1_3(x1_3)], 1))
        x = self.upsample_00(x2_2)
        x = self.chn_aj1(torch.cat([x, x0_0A, x0_0B], dim=1))

        # output1 = self.final1(x0_1)
        # output2 = self.final2(x0_2)
        # output3 = self.final3(x0_3)
        # output4 = self.final4(x0_4)
        output = self.conv_final(x)
        mp = []
        mp.extend(media_p2)
        mp.extend(media_p3)
        mp.extend(media_p4)
        return output, mp