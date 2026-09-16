import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=64, shuffle=True)


def evaluate(test_data, net, device):
    n_correct = 0
    n_total = 0
    net.eval()  # 设置为评估模式
    with torch.no_grad():
        for (x, y) in test_data:
            # 将数据移动到设备
            x, y = x.to(device), y.to(device)
            outputs = net.forward(x.view(-1, 28*28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total


def main():
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net().to(device)  # 将模型移动到设备上
    
    print("初始准确率:", evaluate(test_data, net, device))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # 用于记录训练过程
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(10):
        net.train()  # 设置为训练模式
        epoch_losses = []
        correct = 0
        total = 0
        
        for (x, y) in train_data:
            # 将数据移动到设备
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            output = net.forward(x.view(-1, 28*28))
            loss = torch.nn.functional.nll_loss(output, y)
            loss.backward()
            optimizer.step()

            # 记录损失
            epoch_losses.append(loss.item())
            
            # 计算训练准确率
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        # 计算平均损失和准确率
        avg_loss = np.mean(epoch_losses)
        train_accuracy = correct / total
        test_accuracy = evaluate(test_data, net, device)
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        print(f"Epoch {epoch+1}: 损失={avg_loss:.4f}, 训练准确率={train_accuracy:.4f}, 测试准确率={test_accuracy:.4f}")
    
    # 绘制损失函数下降曲线
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True, alpha=0.3)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'g-', label='Train Accuracy', linewidth=2)
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Test Accuracy Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 预测展示部分 - 需要将数据移回CPU进行显示
    net.eval()  # 设置为评估模式
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        # 将数据移动到设备进行预测
        x_device = x.to(device)
        predict = torch.argmax(net.forward(x_device[0].view(-1, 28*28)))
        
        # 移回CPU进行显示
        plt.figure()
        plt.imshow(x[0].view(28, 28), cmap='gray')
        plt.title(f"Prediction: {int(predict.cpu())}")  # 将预测结果移回CPU
        plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()