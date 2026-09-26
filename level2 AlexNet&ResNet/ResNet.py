import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# ===================== 1. 网络定义 =====================
# 残差块：两层3x3卷积，输出与输入相加（shortcut）
# 如果输入输出通道数或尺寸不一致，用1x1卷积调整shortcut的形状
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # stride不为1或通道数变化时，shortcut需要1x1卷积匹配维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # 残差相加
        out = torch.relu(out)
        return out


# ResNet-18：以CIFAR数据集为例
# 与论文中224输入的版本不同，这里第一层用3x3卷积、去掉最初的MaxPool，
# 保留32x32分辨率到block里再下采样，适合CIFAR10的小图
class MyResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=(2, 2, 2, 2), num_classes=10):
        super(MyResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)   # 32x32
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 16x16
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 8x8
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 4x4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 第一个block负责下采样（用stride），后面的block stride固定为1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ===================== 2. 超参数配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
batch_size = 64
lr = 0.001
epochs = 20

# ===================== 3. 数据预处理 & 数据集 =====================
# ResNet直接在32x32上训练，不需要Resize到224
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ===================== 4. 模型、损失、优化器 =====================
model = MyResNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ===================== 5. 训练+验证循环 =====================
def train_one_epoch(model, loader, criterion, opt, dev):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for img, label in loader:
        img, label = img.to(dev), label.to(dev)
        opt.zero_grad()
        pred = model(img)
        loss = criterion(pred, label)
        loss.backward()
        opt.step()

        total_loss += loss.item()
        _, pred_idx = torch.max(pred.data, 1)
        total += label.size(0)
        correct += (pred_idx == label).sum().item()
    avg_loss = total_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

def val_one_epoch(model, loader, criterion, dev):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(dev), label.to(dev)
            pred = model(img)
            loss = criterion(pred, label)
            total_loss += loss.item()
            _, pred_idx = torch.max(pred.data, 1)
            total += label.size(0)
            correct += (pred_idx == label).sum().item()
    avg_loss = total_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc

if __name__ == "__main__":
    # 每个epoch的指标都记下来，训练完画曲线
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    best_acc = 0
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_one_epoch(model, val_loader, criterion, device)
        t1 = time.time()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch[{epoch+1}/{epochs}] time:{t1-t0:.2f}s")
        print(f"Train Loss:{train_loss:.4f} | Train Acc:{train_acc:.2f}%")
        print(f"Val   Loss:{val_loss:.4f} | Val   Acc:{val_acc:.2f}%\n")

        # 保存最优权重
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "my_resnet_cifar10.pth")
            print(f"保存最优模型, best val acc:{best_acc:.2f}%")

    print("训练结束！")

    # ===================== 6. 画训练曲线 =====================
    epoch_range = range(1, epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epoch_range, history["train_loss"], label="train loss")
    axes[0].plot(epoch_range, history["val_loss"], label="val loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("ResNet18 CIFAR10 Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epoch_range, history["train_acc"], label="train acc")
    axes[1].plot(epoch_range, history["val_acc"], label="val acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy (%)")
    axes[1].set_title("ResNet18 CIFAR10 Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("resnet_cifar10_curve.png", dpi=150)
    print("训练曲线已保存: resnet_cifar10_curve.png")

    # ===================== 7. 加载权重做推理测试 =====================
    test_model = MyResNet(num_classes=10).to(device)
    test_model.load_state_dict(torch.load("my_resnet_cifar10.pth"))
    test_model.eval()

    # 拿一张测试图片预测
    with torch.no_grad():
        test_img, test_label = next(iter(val_loader))
        test_img = test_img[0:1].to(device)  # 取第一张 [1,3,32,32]
        output = test_model(test_img)
        _, pred = torch.max(output, 1)
        print(f"真实标签: {test_label[0].item()}, 预测标签: {pred.item()}")
