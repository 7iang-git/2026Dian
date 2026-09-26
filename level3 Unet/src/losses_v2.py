from __future__ import annotations

"""
改进版损失函数：L1 + Sobel 边缘损失，可选 VGG 感知损失。

动机：baseline 用纯 L1，输出整体误差小但印刷文字的边缘容易发糊。
边缘损失直接对图像梯度做约束，与"保留印刷文字、擦除手写"的目标对齐；
VGG 感知损失把约束从逐像素提升到特征层面，可选开启（需要 torchvision 和预训练权重）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeLoss(nn.Module):
    """
    边缘损失：对预测图和目标图分别做 Sobel 滤波取梯度，
    再计算梯度的 L1 差异。梯度对齐意味着边缘位置和强弱对齐。

    Sobel 核固定不训练。用 groups=in_channels 对 RGB 三通道分别滤波。
    """

    def __init__(self, in_channels=3):
        super(SobelEdgeLoss, self).__init__()

        kernel_x = torch.tensor(
            [
                [-1.0, 0.0, 1.0],
                [-2.0, 0.0, 2.0],
                [-1.0, 0.0, 1.0]
            ],
            dtype=torch.float32
        ).reshape(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)

        kernel_y = torch.tensor(
            [
                [-1.0, -2.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0]
            ],
            dtype=torch.float32
        ).reshape(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)

        # buffer 随模型存取但不参与梯度更新
        self.register_buffer("kernel_x", kernel_x, persistent=False)
        self.register_buffer("kernel_y", kernel_y, persistent=False)
        self.in_channels = in_channels

    def forward(self, predicted, target):
        grad_x_pred = F.conv2d(predicted, self.kernel_x, padding=1, groups=self.in_channels)
        grad_y_pred = F.conv2d(predicted, self.kernel_y, padding=1, groups=self.in_channels)
        grad_x_target = F.conv2d(target, self.kernel_x, padding=1, groups=self.in_channels)
        grad_y_target = F.conv2d(target, self.kernel_y, padding=1, groups=self.in_channels)

        edge_loss = (
            F.l1_loss(grad_x_pred, grad_x_target)
            + F.l1_loss(grad_y_pred, grad_y_target)
        )

        return edge_loss


class VGGPerceptualLoss(nn.Module):
    """
    VGG16 感知损失：取 relu1_2、relu2_2、relu3_3 三层特征做 L1。

    输入是 [0,1] 的图像，先按 ImageNet 的均值方差归一化再进 VGG。
    VGG 全部冻结，只作特征提取。

    需要 torchvision 及 ImageNet 预训练权重（首次运行会自动下载）。
    """

    # VGG16 features 里各层的起始索引：4=ReLU after conv1_2, 9=ReLU after conv2_2, 16=ReLU after conv3_3
    FEATURE_LAYERS = (4, 9, 16)

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, device="cpu"):
        super(VGGPerceptualLoss, self).__init__()

        from torchvision import models

        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        except (AttributeError, TypeError):
            # 旧版本 torchvision 没有 weights 参数
            vgg = models.vgg16(pretrained=True)

        features = list(vgg.features.children())

        # 截取到最后一层用到的位置，之后逐段取特征
        self.blocks = nn.ModuleList()
        start = 0
        for layer_index in self.FEATURE_LAYERS:
            self.blocks.append(nn.Sequential(*features[start:layer_index + 1]))
            start = layer_index + 1

        for parameter in self.parameters():
            parameter.requires_grad = False

        mean = torch.tensor(self.IMAGENET_MEAN).reshape(1, 3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD).reshape(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

        self.eval()
        self.to(device)

    def train(self, mode=True):
        # 感知损失网络永远保持 eval 模式（BN 统计量不能被训练数据更新）
        return super(VGGPerceptualLoss, self).train(False)

    def forward(self, predicted, target):
        predicted = (predicted - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        for block in self.blocks:
            predicted = block(predicted)
            target = block(target)
            loss = loss + F.l1_loss(predicted, target)

        return loss


class CombinedLoss(nn.Module):
    """
    组合损失：total = l1_weight * L1 + edge_weight * Edge [+ vgg_weight * VGG]

    forward 返回 (总损失, 各分量字典)，分量用于训练日志观察。
    use_vgg=False 时 VGG 分量恒为 0。
    """

    def __init__(
        self,
        l1_weight=1.0,
        edge_weight=0.5,
        vgg_weight=0.1,
        use_vgg=False,
        in_channels=3,
        device="cpu"
    ):
        super(CombinedLoss, self).__init__()

        self.l1_weight = l1_weight
        self.edge_weight = edge_weight
        self.vgg_weight = vgg_weight if use_vgg else 0.0
        self.use_vgg = use_vgg

        self.edge_loss = SobelEdgeLoss(in_channels=in_channels).to(device)

        if use_vgg:
            self.vgg_loss = VGGPerceptualLoss(device=device).to(device)

    def forward(self, predicted, target):
        l1 = F.l1_loss(predicted, target)
        edge = self.edge_loss(predicted, target)

        components = {
            "l1": l1.item(),
            "edge": edge.item()
        }

        total = self.l1_weight * l1 + self.edge_weight * edge

        if self.use_vgg:
            vgg = self.vgg_loss(predicted, target)
            components["vgg"] = vgg.item()
            total = total + self.vgg_weight * vgg
        else:
            components["vgg"] = 0.0

        return total, components
