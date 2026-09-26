import torch
import torch.nn as nn
from numpy.ma.core import argmax
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
'''深度学习项目步骤
1 准备数据集
2 搭建神经网路
3 模型训练
4 模型测试
'''

BATCH_SIZE=8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#1 准备数据集
def create_dataset():
    train_data=MNIST(root='../level0 MLP',train=True,transform=ToTensor(),download=True) #[60000,28,28]
    test_data = MNIST(root='../level0 MLP', train=False, transform=ToTensor(), download=True)
    return train_data,test_data

#2 搭建神经网路
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
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1=nn.Conv2d(1,16,3,1,1)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(16,32,3,1,1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.linear1=nn.Linear(1568,64)
        self.linear2=nn.Linear(64,10)

    def forward(self,x):
        #输入的x.shape=[batch,C,H,W]
        x=torch.relu(self.conv1(x))
        x=self.pool1(x)
        x=torch.relu(self.conv2(x))
        x=self.pool2(x) #进行到此处的形状为[batch,32,7,7]
        x=x.view(x.shape[0],-1)
        x=torch.relu(self.linear1(x))
        x=self.linear2(x) #输出的x.shape[batch,10]
        return x
    '''\
    最后要加一层softmax，因为训练的时候损失函数自带了softmax，而推理的时候没有，所以不在此处定义softmax层而是在推理/训练中根据自身情况添加
    '''
#3 模型训练
def train(train_data):
    #3.1 创建数据加载器
    dataloader=DataLoader(train_data,BATCH_SIZE,True)
    #3.2 创建模型对象
    model=MNISTNet().to(device)
    #3.3 创建损失函数对象
    criterion=nn.CrossEntropyLoss() #CrossEntropyLoss 默认对 batch 取平均
    #3.4 创建优化器对象
    optimizer=optim.Adam(model.parameters(),1e-3)
    #3.5 遍历epochs,开始每一轮训练
    epochs=10
    #用于画图的参数
    loss_list = []
    epoch_list = []
    for epoch_idx in range(epochs):
        #定义变量:记录一轮训练下来所有批次的总损失，总样本量（批次数x批次大小），正确预测的样本量，训练开始时间
        total_loss,total_samples,total_correct,start=0.0,0,0,time.time()
    #3.6 遍历dataloader得到每一批次的数据x[batch_size,c,h,w]以及其正确标签y shape:[batch_size]，进行训练
        for x,y in (dataloader):
            x,y=x.to(device),y.to(device)
            #3.6.1 切换模型模式并进行预测
            model.train()
            y_pred=model(x)
            #3.6.2 计算损失
            loss=criterion(y_pred,y)
            #3.6.3 梯度清零（防止上一批次梯度累加） + 反向传播 + 权重更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #3.6.4 累加当前批次预测正确的样本
            total_correct+=(torch.argmax(y_pred,dim=-1)==y).sum()
            #3.6.5 累加当前批次的总损失
            total_loss+=loss.item()*len(y)
            #3.6.6. 累加当前批次的总样本个数
            total_samples+=len(y)
    #3.7 一轮训练结束，打印该轮训练信息,并把平均loss记录进入plt图像中
        print(f'epochs:{epoch_idx+1},loss:{total_loss/total_samples:.5f},accuracy:{total_correct/total_samples:.2f},time:{time.time()-start:.2f}s')
        loss_list.append(total_loss/total_samples)
        epoch_list.append(epoch_idx + 1)
    #3.8  画出loss曲线，保存模型
    torch.save(model.state_dict(),f'./model/mnist.pth')
    plt.figure() #创建画布
    plt.plot(epoch_list, loss_list, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True) #在图中显示网格
    plt.savefig('loss_curve.png')
    plt.show()

#4 模型测试
def predict(test_data):
    #4.1 创建数据加载器
    dataloader=DataLoader(test_data,BATCH_SIZE,False)

    #4.2创建模型对象并加载模型参数
    model=MNISTNet()
    model.load_state_dict(torch.load('./model/mnist.pth'))
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
    # print(f'训练集:{train_data.data.shape}')
    # print(f'测试集:{test_data.data.shape}')
    train(train_data)
    # predict(test_data)
