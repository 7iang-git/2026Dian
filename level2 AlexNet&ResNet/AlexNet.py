import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# ===================== 1. 网络定义 MyAlexNet =====================
class MyAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ===================== 2. 超参数配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
lr = 0.001
epochs = 5

# ===================== 3. 数据预处理 & 数据集 =====================
# CIFAR10原图32×32，AlexNet需要224×224，所以Resize
transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
transform_val = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
val_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ===================== 4. 模型、损失、优化器 =====================
model = MyAlexNet(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ===================== 5. 训练+验证循环 =====================
def train_one_epoch(model, loader, criterion, opt, dev):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (img, label) in enumerate(loader):
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

        # 每50个batch打印一次进度
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch[{batch_idx + 1}/{len(loader)}] | Batch Loss:{loss.item():.4f}")

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
    best_acc = 0
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = val_one_epoch(model, val_loader, criterion, device)
        t1 = time.time()
        print(f"Epoch[{epoch+1}/{epochs}] time:{t1-t0:.2f}s")
        print(f"Train Loss:{train_loss:.4f} | Train Acc:{train_acc:.2f}%")
        print(f"Val   Loss:{val_loss:.4f} | Val   Acc:{val_acc:.2f}%\n")

        # 保存最优权重
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "my_alexnet_cifar10.pth")
            print(f"✅ 保存最优模型, best val acc:{best_acc:.2f}%")

    print("训练结束！")

    # ===================== 6. 加载权重做推理测试 =====================
    # 重新实例化模型
    test_model = MyAlexNet(num_classes=10).to(device)
    test_model.load_state_dict(torch.load("my_alexnet_cifar10.pth"))
    test_model.eval()

    # 拿一张测试图片预测
    with torch.no_grad():
        test_img, test_label = next(iter(val_loader))
        test_img = test_img[0:1].to(device) # 取第一张 [1,3,224,224]
        output = test_model(test_img)
        _, pred = torch.max(output, 1)
        print(f"真实标签: {test_label[0].item()}, 预测标签: {pred.item()}")
