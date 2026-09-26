# Level4: 基于 U-Net 的手写笔记擦除

输入一张带手写答案、批注或演算内容的试卷/作业图片，输出去除手写内容、
尽量保留印刷文字和题目结构的干净图片。对应任务书 Level4。

## 一、数据集

数据来自任务书提供的网盘（20250211 / 20250212 / 20250213 三个批次），
每张原始图对应两张图：

- `input/`：带手写内容的图（模型输入）
- `output/`:擦除手写后的干净图（监督目标 GT）

### 划分

元数据在 `data/metadata/` 下，CSV 格式为 `batch,input_path,output_path,width,height[,split]`：

| 文件 | 样本数 | 用途 |
| :--- | :--- | :--- |
| pairs.csv | 2412 | 全量记录 |
| train.csv | 1930 | 训练（约 80%） |
| val.csv | 241 | 验证（选 best model、早停参考） |
| test.csv | 241 | 测试（只做最终评估） |

按批次划分而不是随机打散，保证同一张图的任何衍生样本不会同时出现在训练和测试中，避免数据泄露。

## 二、数据预处理与增强

实现在 `src/dataset.py`（PairedImageDataset），只依赖 PIL + numpy + torch：

- **train 模式**：input 和 output 先做完全相同的边缘填充，再做同步的随机裁剪得到 256x256，
  这相当于随机平移增强，input/output 用同一个裁剪区域保证像素级对齐；
- **val/test 模式**：短边等比缩放到 256，再边缘填充到 256x256，固定无随机性；
- 像素归一化到 [0, 1]，模型的 Sigmoid 输出与之对应。

## 三、模型结构（src/model.py）

标准 U-Net，base_channels=32（控制参数量，适合 30 元预算）：

```
编码器:  3 -> 32 -> 64 -> 128 -> 256 -> 512   每次 Down 尺寸减半
解码器:  512 -> 256 -> 128 -> 64 -> 32        每次 Up 用 ConvTranspose2d 放大 2 倍
输出:    1x1 Conv -> 3 通道，Sigmoid 到 [0,1]
```

- DoubleConv：两层 3x3 卷积 + BN + ReLU，是每个分辨率级的基本单元；
- Down：MaxPool2d(2) 下采样后接 DoubleConv；
- Up：ConvTranspose2d 上采样 2 倍，与编码器同分辨率的特征图在通道维拼接（跳跃连接）后过 DoubleConv；
- 跳跃连接把编码器保留的高分辨率细节（印刷文字的边缘）直接传给解码器，
  这是 U-Net 适合这种像素级还原任务的关键。

## 四、损失函数

使用 **L1 Loss**，选择理由：

- 任务的目标是逐像素还原，L1 对每个像素的梯度大小恒定（误差符号决定方向），
  不会像 MSE 那样在大误差处梯度过大、小误差处梯度几乎为 0；
- L1 相比 L2 更不容易产生"平均化"的模糊输出，对印刷文字的边缘更友好；
- 验证和测试统一用 L1 + PSNR + SSIM 三个指标衡量（src/metrics.py、src/evaluate.py）。

后续可尝试 L1 + 感知损失（VGG feature loss）或 GAN 损失的组合。

## 五、训练配置（src/train.py）

| 配置 | 值 |
| :--- | :--- |
| 输入尺寸 | 256x256 |
| batch_size | 6 |
| 优化器 | Adam |
| 学习率 | 0.0001 |
| epochs | 40（另有 100 steps 的流程验证小实验） |
| 指标记录 | 每 20 个 step 记录一次 Loss/PSNR/SSIM |
| 随机种子 | 固定（random/numpy/torch 全部设置） |
| 训练环境 | AutoDL GPU 服务器 |
| 服务器费用 | 1h 3070ti 1.08元 |

## 六、实验结果

### pilot_100steps（流程验证）

正式训练前用 100 steps 的小实验验证整条流程（数据-训练-评估-可视化）能跑通，符合任务书
"训练前先用小数据和少量 Epoch 验证流程"的要求。

### experiment_50epoch（正式训练，实际 40 epoch）

| 指标 | best model (epoch 37) | final (epoch 40) |
| :--- | :--- | :--- |
| 测试集 PSNR | 20.19 | 20.58 |
| 测试集 SSIM | 0.8560 | 0.8528 |

- Loss / PSNR / SSIM 曲线：`outputs/experiment_50epoch/step_metrics.png`（step 级）和
  `epoch_metrics.png`（epoch 级）；
- 详细数值：同目录下 `loss.csv`、`epoch_metrics.csv`、`step_metrics.csv`、`test_metrics.csv`；
- 可视化对比图（input / prediction / GT 三联）：`outputs/experiment_50epoch/visualize/samples/`，
  肉眼检查大部分样本手写内容被擦除、印刷结构保留，没有出现全白全黑；
- 最终权重：`outputs/experiment_50epoch/checkpoints/best_model.pth`。

### experiment_v2_50epoch（改进版 UNetV2 + L1 + 边缘损失，本地 RTX 5070 训练 40 epoch）

| 指标 | best model (epoch 12) | final (epoch 40) |
| :--- | :--- | :--- |
| 测试集 PSNR | 21.32 | 20.73 |
| 测试集 SSIM | 0.8711 | 0.8574 |

- 曲线和详细数值：`outputs/experiment_v2_50epoch/` 下同名文件，
  step_metrics.csv 额外记录 l1/edge/vgg 各损失分量；
- best model 可视化：`outputs/experiment_v2_50epoch/visualize/best_model_samples/`。

对比结论：

- 改进版 best model 相比 baseline best model，PSNR +1.13 dB、SSIM +0.015，
  残差块 + 注意力门控 + 边缘损失的组合有效；
- 两个实验的 best model 都出现在训练中前期（epoch 37 / 12），之后验证集
  SSIM 不再上升，说明在 1930 张训练图的数据量下模型收敛很快，继续训练
  在过拟合。后续方向是增强数据增强强度或加正则，而不是加长训练；
- 可视化注意：可视化必须用 best model 生成。train_v2.py 已修正为保存
  可视化前先加载 best checkpoint（早期版本误用最终 epoch 的模型，
  过拟合的最终模型视觉效果明显偏差）。

## 七、目录结构

```
level3 Unet/
  data/                  数据集与 metadata CSV
  src/
    dataset.py           PairedImageDataset 成对数据加载与增强
    model.py             UNet 定义
    train.py             训练主脚本（含验证/测试/曲线/可视化）
    evaluate.py          独立评估脚本
    metrics.py           PSNR / SSIM 计算
    visualize.py         结果可视化
  outputs/               训练输出（checkpoints、曲线、CSV、可视化）
  train.log              服务器训练日志
  requirement.txt        环境依赖
```

## 八、如何运行

### 1. 环境准备

服务器（AutoDL，Python 3.11 + CUDA）和本地均可。先装 PyTorch 的 CUDA 版本
（按服务器的 CUDA 版本选择 index-url），再装其余依赖：

```bash
# 以 CUDA 12.1 为例，其他版本见 pytorch.org
pip install torch --index-url https://download.pytorch.org/whl/cu121

pip install -r requirement.txt
```

验证 GPU 可用：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

输出 True 即可正常训练（False 也能跑，但会慢很多）。

### 2. 数据准备

数据集从任务书提供的网盘链接下载，解压后按下面的目录结构放到
`level3 Unet/data/` 下：

```
level3 Unet/
  data/
    20250211/
      dataset/
        input/    带手写内容的图（模型输入）
        output/   擦除手写后的干净图（GT）
    20250212/
      ...
    20250213/
      ...
    metadata/
      pairs.csv   全量配对记录（2412 对）
      train.csv   训练集（1930）
      val.csv     验证集（241）
      test.csv    测试集（241）
```

CSV 格式为 `batch,input_path,output_path,width,height[,split]`，
图片路径是相对 `data/` 的路径，例如 `20250211/dataset/input/xxx.jpg`。

### 3. 训练 baseline（原版 UNet + L1）

```bash
cd src
python train.py
```

- 结果输出到 `outputs/` 下 `train.py` 里 `OUTPUT_DIR` 常量指定的目录
  （当前为 `experiment_50epoch`）；
- 做新实验时改这个常量的名字，旧实验结果就不会被覆盖。

### 4. 训练改进版（UNetV2 + 组合损失）

```bash
cd src
python train_v2.py
```

- 结果输出到 `outputs/experiment_v2_50epoch/`，与 baseline 互不干扰；
- 训练前可以在文件顶部调参数：

| 常量 | 默认值 | 说明 |
| :--- | :--- | :--- |
| NUM_EPOCHS | 40 | 正式训练轮数，想快速验证流程可先改成 1 |
| EDGE_WEIGHT | 0.5 | Sobel 边缘损失权重 |
| USE_VGG_LOSS | False | 开启 VGG 感知损失需设 True，并安装 torchvision |
| VGG_WEIGHT | 0.1 | VGG 感知损失权重 |
| MAX_TRAIN_BATCHES | None | 调试时可设成小整数，只用少量 batch 跑通流程 |

### 5. 评估与可视化

训练脚本结束时会自动完成：
- 用 best model 和 final model 分别评估测试集（`test_metrics.csv`）；
- 输出 Loss/PSNR/SSIM 曲线（`step_metrics.png`、`epoch_metrics.png`）；
- 保存 input / prediction / GT 三联对比图（`visualize/samples/`）。

如需对已有 checkpoint 单独评估或可视化，用这两个脚本（它们从文件顶部
的常量读配置，运行前先把 `OUTPUT_DIR`、`CHECKPOINT_PATH` 等改成目标实验的路径）：

```bash
python evaluate.py    # 对指定 checkpoint 跑测试集指标
python visualize.py   # 对指定 checkpoint 生成对比图
```

注意：这两个脚本目前 `from model import UNet`，评估 UNetV2 的 checkpoint
需要把这行换成 `from model_v2 import UNetV2` 并同步改实例化处。

### 6. 输出文件说明

每个实验目录（如 `outputs/experiment_v2_50epoch/`）下的文件：

| 文件 | 内容 |
| :--- | :--- |
| loss.csv | 每个 epoch 的 train/val loss 和验证 PSNR/SSIM |
| epoch_metrics.csv / .png | epoch 级指标及曲线 |
| step_metrics.csv / .png | 每 20 step 的训练指标及曲线（v2 额外含 l1/edge/vgg 分量） |
| test_metrics.csv | best model 和 final model 的测试集指标 |
| checkpoints/ | 每个 epoch 的权重 + best_model.pth |
| visualize/samples/ | input / prediction / GT 三联对比图 |

### 7. 对比 baseline 与改进版

分别取 `outputs/experiment_50epoch/test_metrics.csv` 和
`outputs/experiment_v2_50epoch/test_metrics.csv`，把 PSNR/SSIM 填进
README 第三节的实验结果表。训练曲线也建议并排截图放进 README，
作为"改进是否有效"的实验证据。

## 九、改进方案详解（任务书要求思考）

### 9.1 改进动机

baseline（原版 UNet + 纯 L1）跑出 PSNR 20.58 / SSIM 0.8528 后，检查可视化
发现两个问题：

1. **边缘发糊**：L1 是逐像素平均约束，误差被摊到每个像素上，印刷文字的
   笔画边缘容易出现灰色的过渡带；
2. **跳跃连接无差别传递**：编码器特征里既有印刷文字的细节，也有噪声和
   与当前解码位置无关的响应，全部送进解码器反而干扰融合。

针对这两点分别改进模型和损失，同时保持训练成本可控（30 元预算）。

### 9.2 模型改进：UNetV2（src/model_v2.py）

改动 1：DoubleConv 换成 ResidualDoubleConv（残差块）

```
baseline:  x -> Conv-BN-ReLU -> Conv-BN -> out
改进版:    x -> Conv-BN-ReLU -> Conv-BN -> (+) -> ReLU -> out
           └-------- 1x1 Conv-BN（通道变化时） --------┘
```

- 残差相加让每层学"修正量"而不是完整映射，梯度可以走 shortcut 直接回传
  （思路来自 ResNet，level2 的 README 里有详细笔记）；
- 通道数变化时 shortcut 用 1x1 卷积对齐维度。

改动 2：跳跃连接加 Attention Gate（加性注意力，来自 Attention U-Net）

```
skip(编码器特征) ── 1x1Conv-BN ──┐
                                 (+) -> ReLU -> 1x1Conv-BN-Sigmoid -> 注意力图 a
gate(解码器特征) ─ 1x1Conv-BN ──┘
输出 = skip * a        （a 在 0-1 之间，逐像素抑制无关响应）
```

- gate 是解码器一侧上采样后的粗特征，语义强但分辨率低；
- 它与 skip 特征做加性交互生成注意力图，"这一块和当前解码位置有关吗"
  无关的位置被压到接近 0，印刷文字所在的边缘区域被保留；
- 相比原始 UNet 的直接拼接，等于给跳跃连接加了一个可学习的过滤器。

保持公平对比的措施：拓扑（4 次下采样、每级 2 层卷积）、base_channels=32、
Sigmoid 输出、接口签名都与 baseline 完全一致。参数量从 7.77M 增加到
9.81M（+26.4%，增量来自残差 shortcut 和注意力门控的 1x1 卷积），
同 epoch 数下训练耗时增幅相近。

### 9.3 损失改进：CombinedLoss（src/losses_v2.py）

总损失公式（训练用）：

```
total = 1.0 * L1(pred, target)
      + 0.5 * SobelEdge(pred, target)
      [+ 0.1 * VGGPerceptual(pred, target)   可选，默认关闭]
```

**Sobel 边缘损失**：用固定（不训练）的 Sobel 核分别提取预测图和目标图的
水平/垂直梯度，对梯度做 L1：

- 像素差对齐的是"颜色"，梯度差对齐的是"边缘的位置和强弱"；
- 手写擦除任务里印刷文字的笔画边缘是最需要保住的结构，直接对梯度加约束
  与这个目标一致；
- 本地验证过有效性：清晰输出与 GT 的 edge loss 为 0，模糊后的输出升到 1.28，
  说明该损失确实能"感觉到"模糊。

**VGG 感知损失（可选）**：冻结的 VGG16 提取 relu1_2/relu2_2/relu3_3 三层特征，
各层做 L1 后求和。逐像素损失衡量"每个像素对不对"，感知损失衡量
"看起来像不像"，对整体结构的还原有帮助。默认关闭（需要 torchvision 和
ImageNet 预训练权重），开启方式见第八节参数表。

### 9.4 训练脚本差异（src/train_v2.py）

| 项目 | train.py（baseline） | train_v2.py（改进版） |
| :--- | :--- | :--- |
| 模型 | UNet | UNetV2 |
| 训练损失 | L1 | L1 + Edge (+ VGG) |
| 评估损失 | L1 | L1（刻意与 baseline 口径一致） |
| 输出目录 | outputs/experiment_50epoch | outputs/experiment_v2_50epoch |
| step CSV 字段 | loss/psnr/ssim | 额外含 l1/edge/vgg 分量 |
| checkpoint 键 | model_state_dict | 同左，另加 "model": "UNetV2" 标识 |

数据加载、指标计算（PSNR/SSIM）、best model 选择标准（验证集 SSIM）、
曲线绘制、三联可视化全部沿用 baseline 的实现，只换模型和损失，
保证差异全部来自改进本身而不是流程变化。

### 9.5 预期效果与验证方法

改进后预期：PSNR/SSIM 与 baseline 持平或略升，且预测图的笔画边缘更锐利
（visualize 三联图里对比笔画是否有灰色过渡带）；step_metrics.csv 里 edge
分量应随训练明显下降。验证方式：对比两个实验的 test_metrics.csv 和可视化图，
结论写回第三节实验结果表。

### 9.6 后续方向（未实现）

1. **GAN**：加一个 patch-level 判别器（类似 pix2pix），对抗损失能让输出更"像"真实的干净试卷，
   缓解 L1 输出偏模糊的问题，但训练不稳定，需要控制权重比例；
2. **Diffusion Model**：把擦除建模为条件去噪（如 palette / DDPM 类图像修复思路），
   质量上限更高，但推理要几十步采样，成本和速度是权衡点；
3. **数据/输入改进**：检测手写区域生成 mask，把"擦除"拆成"检测 + 局部修复"两阶段，
   或者直接做 inpainting（只重绘手写区域），避免对整图重生成引入印刷文字的失真。
