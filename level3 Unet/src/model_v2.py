from __future__ import annotations

"""
level3 Unet 改进版模型：ResUNet + Attention Gate。

相对 baseline UNet（model.py）的改动：
1. DoubleConv 换成 ResidualDoubleConv：每个分辨率级内部带残差连接，
   深一点的编码器也能稳定训练；
2. 跳跃连接上加入 Attention Gate（Attention U-Net, Oktay et al. 2018）：
   解码器的粗特征作为 gate，对编码器的 skip 特征做逐像素筛选，
   抑制与当前解码位置无关的背景响应，让印刷文字的边缘信息更容易通过；
3. 拓扑与 baseline 完全一致（4 次下采样、base_channels=32），
   参数量增幅很小，方便和 baseline 公平对比。

对外接口与 baseline 相同：UNetV2(in_channels=3, out_channels=3, base_channels=32)，
输入 [B,3,256,256]，输出 Sigmoid 到 [0,1]。
"""

import torch
import torch.nn as nn


class ResidualDoubleConv(nn.Module):
    """两层 3x3 卷积 + BN，带残差相加。通道数变化时 shortcut 用 1x1 卷积对齐。"""

    def __init__(self, in_channels, out_channels):
        super(ResidualDoubleConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


class AttentionGate(nn.Module):
    """
    加性注意力门控。

    gate 是解码器一侧的粗特征（语义强、分辨率低），
    x 是编码器一侧的 skip 特征（分辨率高、细节多）。
    两者各过 1x1 卷积后相加，生成 0-1 的注意力图再乘回 skip 特征。
    """

    def __init__(self, gate_channels, skip_channels, inter_channels):
        super(AttentionGate, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        attention = self.psi(self.relu(self.W_gate(gate) + self.W_skip(skip)))
        return skip * attention


class Down(nn.Module):
    """MaxPool 下采样一半，再接残差卷积块。"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ResidualDoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    """
    转置卷积上采样 2 倍，然后：
    1. 以上采样特征为 gate，对编码器 skip 特征做注意力筛选；
    2. 拼接后过残差卷积块。
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.attention = AttentionGate(
            gate_channels=in_channels,
            skip_channels=skip_channels,
            inter_channels=out_channels // 2
        )
        self.block = ResidualDoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # 尺寸差 1 个像素时（奇数输入）补齐，256 输入下不会触发
        if x.shape[-1] != skip.shape[-1] or x.shape[-2] != skip.shape[-2]:
            x = nn.functional.pad(
                x,
                (0, skip.shape[-1] - x.shape[-1], 0, skip.shape[-2] - x.shape[-2])
            )

        skip = self.attention(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class UNetV2(nn.Module):
    """
    改进版 U-Net。

    通道数变化（base_channels=32）：
        编码器:  3 -> 32 -> 64 -> 128 -> 256 -> 512（瓶颈层）
        解码器:  512 -> 256 -> 128 -> 64 -> 32
        输出:    32 -> 3
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super(UNetV2, self).__init__()

        self.input_conv = ResidualDoubleConv(in_channels, base_channels)

        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.up1 = Up(base_channels * 16, base_channels * 8, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels, base_channels)

        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self.output_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.input_conv(x)   # [B,  32, 256, 256]
        x2 = self.down1(x1)       # [B,  64, 128, 128]
        x3 = self.down2(x2)       # [B, 128,  64,  64]
        x4 = self.down3(x3)       # [B, 256,  32,  32]
        x5 = self.down4(x4)       # [B, 512,  16,  16]

        x = self.up1(x5, x4)      # [B, 256,  32,  32]
        x = self.up2(x, x3)       # [B, 128,  64,  64]
        x = self.up3(x, x2)       # [B,  64, 128, 128]
        x = self.up4(x, x1)       # [B,  32, 256, 256]

        return self.output_sigmoid(self.output_conv(x))
