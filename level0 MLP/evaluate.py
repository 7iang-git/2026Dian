'''
用于单张图片推理的脚本
'''
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# ============ 1. 定义和训练时完全一致的模型结构 ============
class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

# ============ 2. 加载模型 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(device)
model.load_state_dict(torch.load('./model/mnist_model.pth', weights_only=False,map_location=device))
model.eval()          # 关闭 dropout/batchnorm 的训练行为

# ============ 3. 单张图片预处理 ============
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # 确保单通道
    transforms.Resize((28, 28)),                   # 统一尺寸
    transforms.ToTensor(),                         # → [0,1] 张量, shape=[1,28,28]
    transforms.Normalize((0.1307,), (0.3081,)),    # 与训练一致
])

# ============ 4. 推理函数 ============
@torch.no_grad()
def predict(image_path):
    img = Image.open(image_path).convert("L")      # 转灰度
    x = transform(img).unsqueeze(0).to(device)     # [1,1,28,28] 加 batch 维

    logits = model(x.view(x.size(0), -1))                              # [1,10]
    probs  = torch.softmax(logits, dim=1)
    pred   = probs.argmax(dim=1).item()
    conf   = probs.max().item()

    return pred, conf, probs.squeeze().cpu()

# ============ 5. 运行 ============
if __name__ == "__main__":
    img_path = "./MNIST/test/2/2_0000.png"
    pred, conf, probs = predict(img_path)

    print(f"预测结果: {pred}  置信度: {conf:.4f}")
    print("各类概率:", [f"{p:.3f}" for p in probs])

    # 可视化
    img = Image.open(img_path).convert("L")
    plt.imshow(img, cmap="gray")
    plt.title(f"Pred: {pred} ({conf:.2%})")
    plt.axis("off")
    plt.show()