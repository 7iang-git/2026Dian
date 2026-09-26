# ============================================================
# U-Net 用于成对图像到手写去除（image-to-image）
#
# 整体结构：
#   编码器（下采样）：4 次，每次空间尺寸减半，通道数翻倍
#   瓶颈层：最底层特征
#   解码器（上采样）：4 次，每次空间尺寸翻倍，并与编码器对应层拼接
#   输出层：1x1 卷积 + Sigmoid，输出 RGB 图像，像素范围 [0, 1]
#
# 输入:  [B, 3, H, W]，像素值 [0, 1]
# 输出:  [B, 3, H, W]，像素值 [0, 1]
# 要求 H、W 是 16 的倍数（如 256），否则解码时尺寸可能对不上
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    双层卷积模块（U-Net 的基本单元）：
        Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    因为 kernel_size=3, padding=1，所以输入输出空间尺寸不变，
    只改变通道数。
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        # nn.Sequential 会按顺序依次执行里面的层
        self.layers = nn.Sequential(
            # 第 1 个 3x3 卷积: in_channels -> out_channels
            # padding=1 保证 H、W 不变
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            # 批归一化：加速收敛、稳定训练
            nn.BatchNorm2d(out_channels),
            # ReLU 激活，inplace=True 节省显存
            nn.ReLU(inplace=True),

            # 第 2 个 3x3 卷积: out_channels -> out_channels
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 直接把输入喂给 Sequential，返回处理后的特征图
        return self.layers(x)


class Down(nn.Module):
    """
    下采样模块：
        MaxPool2d(2)  ->  空间尺寸减半
        DoubleConv    ->  通道数变换
    """

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.layers = nn.Sequential(
            # 2x2 最大池化，stride 默认等于 kernel_size=2，尺寸减半
            nn.MaxPool2d(2),
            # 池化后再做双层卷积，改变通道数
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):
    """
    上采样模块：
        1. ConvTranspose2d 把空间尺寸放大 2 倍
        2. 与编码器对应层的特征图在通道维拼接（skip connection）
        3. 再经过 DoubleConv 融合特征

    参数:
        in_channels:   输入特征图通道数（来自上一层解码器或瓶颈层）
        skip_channels: 编码器对应层特征图通道数（用于拼接）
        out_channels:  输出特征图通道数
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(Up, self).__init__()

        # 转置卷积（反卷积）：kernel_size=2, stride=2
        # 效果：H、W 各放大 2 倍，通道数 in_channels -> out_channels
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        # 拼接后通道数 = out_channels（上采样结果） + skip_channels（编码器特征）
        # 输出通道数 = out_channels
        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        # 1) 上采样：空间尺寸扩大 2 倍
        x = self.up(x)

        # 2) 尺寸对齐
        # 当输入尺寸不是 2 的整数次幂时，上采样结果和 skip 可能差 1 个像素
        # 这里用 F.pad 补零对齐，保证后面 torch.cat 能正常拼接
        diff_y = skip.size(2) - x.size(2)   # 高度差
        diff_x = skip.size(3) - x.size(3)   # 宽度差

        # F.pad 的填充顺序: (左, 右, 上, 下)
        # 左边/上边补 diff//2，右边/下边补剩余部分，尽量均匀
        x = F.pad(
            x,
            (
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2
            )
        )

        # 3) 通道维拼接：dim=1 表示沿着通道方向
        # skip: [B, skip_channels, H, W]
        # x:    [B, out_channels,  H, W]
        # 拼接后: [B, skip_channels + out_channels, H, W]
        x = torch.cat([skip, x], dim=1)

        # 4) 卷积融合拼接后的特征
        x = self.conv(x)

        return x


class UNet(nn.Module):
    """
    U-Net 主体。

    通道数随层数变化（以 base_channels=32 为例）：
        编码器:  3 -> 32 -> 64 -> 128 -> 256 -> 512（瓶颈层）
        解码器:  512 -> 256 -> 128 -> 64 -> 32
        输出:    32 -> 3
    """

    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super(UNet, self).__init__()

        # ========================================================
        # 编码器（下采样路径）
        # ========================================================

        # 第 1 层：输入 -> base_channels（如 3 -> 32），尺寸不变
        self.input_conv = DoubleConv(in_channels, base_channels)

        # 第 2 层：base_channels -> base_channels*2（如 32 -> 64），尺寸减半
        self.down1 = Down(base_channels, base_channels * 2)

        # 第 3 层：base_channels*2 -> base_channels*4（如 64 -> 128），尺寸再减半
        self.down2 = Down(base_channels * 2, base_channels * 4)

        # 第 4 层：base_channels*4 -> base_channels*8（如 128 -> 256），尺寸再减半
        self.down3 = Down(base_channels * 4, base_channels * 8)

        # 第 5 层（瓶颈层）：base_channels*8 -> base_channels*16（如 256 -> 512），尺寸再减半
        self.down4 = Down(base_channels * 8, base_channels * 16)

        # ========================================================
        # 解码器（上采样路径）
        # 参数顺序: (输入通道, skip 通道, 输出通道)
        # ========================================================

        # up1: 瓶颈层 (16x) 上采样后，与 down3 输出 (8x) 拼接 -> 8x
        self.up1 = Up(
            base_channels * 16,
            base_channels * 8,
            base_channels * 8
        )

        # up2: 上一级 (8x) 上采样后，与 down2 输出 (4x) 拼接 -> 4x
        self.up2 = Up(
            base_channels * 8,
            base_channels * 4,
            base_channels * 4
        )

        # up3: 上一级 (4x) 上采样后，与 down1 输出 (2x) 拼接 -> 2x
        self.up3 = Up(
            base_channels * 4,
            base_channels * 2,
            base_channels * 2
        )

        # up4: 上一级 (2x) 上采样后，与 input_conv 输出 (1x) 拼接 -> 1x
        self.up4 = Up(
            base_channels * 2,
            base_channels,
            base_channels
        )

        # ========================================================
        # 输出层
        # ========================================================

        # 1x1 卷积：把 base_channels 压到 out_channels（如 32 -> 3）
        # 1x1 卷积只改变通道数，不改变空间尺寸
        self.output_conv = nn.Conv2d(
            base_channels,
            out_channels,
            kernel_size=1
        )

        # Sigmoid 把输出压到 [0, 1]，与训练数据的归一化方式对应
        self.output_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # ========================================================
        # 编码器前向：逐步下采样
        # 假设输入 256x256，特征图尺寸变化如下：
        #   x1: [B,  32, 256, 256]
        #   x2: [B,  64, 128, 128]
        #   x3: [B, 128,  64,  64]
        #   x4: [B, 256,  32,  32]
        #   x5: [B, 512,  16,  16]
        # ========================================================
        x1 = self.input_conv(x)   # 第 1 层，尺寸不变
        x2 = self.down1(x1)       # 第 2 层，尺寸减半
        x3 = self.down2(x2)       # 第 3 层，尺寸再减半
        x4 = self.down3(x3)       # 第 4 层，尺寸再减半
        x5 = self.down4(x4)       # 瓶颈层，尺寸再减半

        # ========================================================
        # 解码器前向：逐步上采样，并与编码器对应层拼接
        #   up1: x5 (16x16) 上采样 -> 32x32，与 x4 拼接
        #   up2: 上一步 (32x32) 上采样 -> 64x64，与 x3 拼接
        #   up3: 上一步 (64x64) 上采样 -> 128x128，与 x2 拼接
        #   up4: 上一步 (128x128) 上采样 -> 256x256，与 x1 拼接
        # ========================================================
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # ========================================================
        # 输出层
        # 1x1 卷积: [B, 32, 256, 256] -> [B, 3, 256, 256]
        # Sigmoid:  把像素值压到 [0, 1]
        # ========================================================
        x = self.output_conv(x)
        x = self.output_sigmoid(x)

        return x

