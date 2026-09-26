import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import csv
import os

# 加载训练时保存的最优权重，在 CIFAR-10 测试集上评估准确率。
# 结果写入 results.csv，README 里的数字以这份结果为准。

import AlexNet
import ResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

transform_32 = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

test_set_224 = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_224)
test_set_32 = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_32)

def evaluate(model, weights_path, dataset, input_desc):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            pred = model(img)
            _, pred_idx = torch.max(pred, 1)
            correct += (pred_idx == label).sum().item()
            total += label.size(0)

    acc = 100.0 * correct / total
    print("{} ({}): {:.2f}%  ({}/{})".format(weights_path, input_desc, acc, correct, total))
    return acc

if __name__ == "__main__":
    results = []

    acc_alexnet = evaluate(
        AlexNet.MyAlexNet(num_classes=10),
        "my_alexnet_cifar10.pth",
        test_set_224,
        "224x224"
    )
    results.append(["AlexNet", "my_alexnet_cifar10.pth", "224x224", "{:.2f}%".format(acc_alexnet)])

    acc_resnet = evaluate(
        ResNet.MyResNet(num_classes=10),
        "my_resnet_cifar10.pth",
        test_set_32,
        "32x32"
    )
    results.append(["ResNet-18", "my_resnet_cifar10.pth", "32x32", "{:.2f}%".format(acc_resnet)])

    with open("results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "weights", "input_size", "test_accuracy"])
        writer.writerows(results)

    print("结果已保存: results.csv")
