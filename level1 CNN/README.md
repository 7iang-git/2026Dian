## 实验概述

本实验复用 Level1 的 MNIST 数据集，将模型替换为卷积神经网络 CNN，保持相近训练流程。从**准确率、参数量、收敛速度、错误样本**四个维度对比 CNN 与 MLP，验证卷积网络针对图像任务的先验优势。

## 环境与数据集

- 运行环境：Anaconda，Python=3.11.14，Pytorch=2.7.1+cu128，CUDA=12.8
- 数据集：MNIST 手写数字数据集
- 数据集划分：训练集 60000 张，测试集 10000 张；无训练集数据泄露
- 预处理操作：
	1. `ToTensor()`，转为 [1,28,28] 张量
	2. 图像保留二维空间结构，**不做展平操作**

## 模型结构

### 1. CNN 网络结构（MNISTNet）

```
Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1) → ReLU
MaxPool2d(kernel_size=2, stride=2)                # 28 → 14
Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) → ReLU
MaxPool2d(kernel_size=2, stride=2)                # 14 →7
Flatten()                                          # shape: [batch, 32*7*7 =1568]
Linear(1568, 64) → ReLU
Linear(64, 10)                                     # 输出10分类logits
```

- 总参数量：**105866**

> 计算过程：conv1 (160) + conv2 (4640) + linear1 (100416) + linear2 (650) = 105866

### 2. 对照 MLP 模型（来自 Level‑1）

```
Flatten(28*28=784) → Linear(784, hidden1) → ReLU → Linear(hidden1, hidden2) → ReLU → Linear(hidden2,10)
```

> MLP总参数**59,210**

## 训练超参数

表格

| 超参数     | 取值                  |
| ---------- | --------------------- |
| Batch Size | 8                     |
| 学习率 lr  | 1e‑3                  |
| Epoch 数量 | 10                    |
| 优化器     | Adam                  |
| 损失函数   | CrossEntropyLoss      |
| 设备       | cuda /cpu（自动选择） |

## 实验结果

| 指标         | CNN    | MLP(Level‑1) |
| ------------ | :----- | ------------ |
| 测试集准确率 | 0.99   | 0.97         |
| 模型总参数量 | 105866 | 59,210       |

### 训练曲线说明

- 附件：`level1 CNN/train_accuracy_curve.png,level1 CNN/loss_curve.png` CNN 训练 & 验证 loss、准确率曲线
- 附件：`level0 MLP/result/损失曲线与准确率曲线.png` MLP 训练 loss、准确率曲线



## 实验分析与思考（对应题目思考题）

1. MLP 需要把图像展平为一维向量，完全丢失像素的空间位置、相邻关系；每一个像素和下一层神经元是全连接，没有对 “局部相邻像素” 做优先提取，参数量巨大。
2. CNN 使用卷积核滑动，**局部感受野、权值共享**：天然假设图像有效特征来自局部相邻像素，这就是图像任务的先验约束。
3. 权值共享大幅降低参数量；池化带来一定平移鲁棒性；所以 CNN 收敛更快、泛化能力更强，图像分类效果优于 MLP。
4. 卷积核大小影响：小卷积核堆叠可以获得更大感受野同时参数量更小；大卷积核参数量大，容易过拟合。

## 复现说明

1. 训练和推理脚本：`cnn.py`；在main函数中执行train/predict函数可切换
2. 运行命令示例

```
python cnn.py
```