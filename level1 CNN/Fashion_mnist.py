from torchvision.datasets import FashionMNIST
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import time
import torch.optim as optim

'''
CNN网络识别fashion_mnist数据集
'''
BATCH_SIZE=8

def create_dataset():
    train_data=FashionMNIST('./data',True,transform=ToTensor(),download=True)
    test_data=FashionMNIST('./data',False,transform=ToTensor(),download=True)
    return train_data,test_data

class FashionNet(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        输入：[batch, 1, 28, 28]
        ├─ Conv2d( in_channels=1, out_channels=16, kernel_size=3, padding=1 )
        ├─ ReLU
        ├─ MaxPool2d( kernel_size=2, stride=2 )    # 尺寸从28→14
        ├─ Conv2d( in_channels=16, out_channels=32, kernel_size=3, padding=1 )
        ├─ ReLU
        ├─ MaxPool2d( kernel_size=2, stride=2 )    # 尺寸从14→7
        ├─ Flatten()                               # 展平：[batch,32*7*7=1568]
        ├─ Linear(1568, 64)
        ├─ ReLU
        └─ Linear(64, 10) 输出：[batch,10]→ log_softmax
        '''
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(1568, 64)
        self.linear2 = nn.Linear(64, 10)

    def forward(self, x):
        # 输入的x.shape=[batch,C,H,W]
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)  # 进行到此处的形状为[batch,32,7,7]
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)  # 输出的x.shape[batch,10]
        return x

def train(train_data):
    dataloader=DataLoader(train_data,BATCH_SIZE,True)
    model=FashionNet().to(device='cuda')
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),1e-3)

    epochs=10
    loss_list = []
    epoch_list = []
    model.train()
    for epochs_idx in range(epochs):
        total_loss,total_samples,total_correct,start=0.0,0,0,time.time()
        for x,y in dataloader:
            x=x.to(device='cuda')
            y=y.to(device='cuda')
            y_pred=model(x)
            loss=criterion(y_pred,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() #更新参数
            total_samples+=len(y)
            total_loss+=loss*len(y)
            total_correct+=(torch.argmax(y_pred,-1)==y).sum()
        print(f'epochs:{epochs_idx+1},loss:{total_loss/total_samples:.5f},accuracy:{total_correct/total_samples:.2f},time:{time.time()-start:.2f}')
        loss_list.append(total_loss.item() / total_samples)
        epoch_list.append(epochs_idx + 1)
    torch.save(model.state_dict(),'./model/fashion_mnist.pth')
    plt.figure()  # 创建画布
    plt.plot(epoch_list, loss_list, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)  # 在图中显示网格
    plt.savefig('loss_curve.png')
    plt.show()

def predict(test_data):
    #4.1 创建数据加载器
    dataloader=DataLoader(test_data,BATCH_SIZE,False)

    #4.2创建模型对象并加载模型参数
    model=FashionNet()
    model.load_state_dict(torch.load('./model/fashion_mnist.pth'))
    total_correct,total_samples=0,0

    #4.3 切换模型模式开始推理
    model.eval()
    for x,y in dataloader:
        y_pred=model(x)
        #由于这里只需要判断概率最大的类别而不需要输出每个类别的概率，所以不加softmax，用argmax代替
        y_pred=torch.argmax(y_pred,dim=-1)
    #4.4 统计正确率
        total_correct+=(y_pred==y).sum()
        total_samples+=len(y)
    print(f'accuracy:{total_correct/total_samples:.2f}')

if __name__=='__main__':
    train_data,test_data=create_dataset()
    # print(f'{train_data.data.shape}')
    # train(train_data)
    predict(test_data)
