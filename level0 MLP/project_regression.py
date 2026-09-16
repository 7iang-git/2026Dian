import torch

#输入数据的转化过程：numpy->tensor->Dataset->DataLoader 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch import nn   
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt 

plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False

#1 生成数据集

#这个生成的是numpy数组  
x,y,coef=make_regression(n_samples=100,n_features=1,bias=14.5,noise=10,coef=True,random_state=3)
plt.scatter(x,y,label="原始数据")
plt.legend()
plt.show()
#转换为tensor
x=torch.tensor(x,dtype=torch.float32)
y=torch.tensor(y,dtype=torch.float32)
#将输入数据和输出数据合并为一个数据集Dataset
dataset=torch.utils.data.TensorDataset(x,y)
#将数据集分成多个批次 也就是DataLoader
dataloader=DataLoader(dataset,batch_size=16,shuffle=True)

#2 模型训练
def train(x,y,coef):
    # 创建模型
    #参1 输入特征数 参2 输出特征数
    model=nn.Linear(1,1)
    # 创建损失函数  这个的底层是输入一批次数据会输出该批次的平均损失
    criterion=nn.MSELoss()
    # 创建优化器
    optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
    # 训练模型
        #1 创建变量分别表示 轮数 每轮平均损失  总损失 训练样本批次数
    epochs=100
    loss_list=[]
    total_loss=0.0
    num_train_samples=0
        #2 轮次循环
    for epoch in range(epochs):
        #3 批次循环
        for batch_x,batch_y in dataloader:
        #4 正向传播
            pred=model(batch_x)
        #5 计算每一批的(平均)损失
            loss=criterion(pred,batch_y.reshape(-1,1))
        #6 记录损失
            total_loss+=loss.item()
            num_train_samples+=1
        #7 反向传播
            optimizer.zero_grad()
            loss.sum().backward()
        #8 更新参数
            optimizer.step()
        #9 计算每轮平均损失
        avg_loss=total_loss/num_train_samples
        loss_list.append(avg_loss)
        #10 打印每一轮损失
        print(f"轮次:{epoch+1} 平均损失:{avg_loss:.4f}")
    #打印最终训练成果
    print(f"模型参数w:{model.weight.item():.4f} b:{model.bias.item():.4f}")
    #可视化损失曲线
    plt.plot (range (epochs), loss_list)
    plt.title("训练损失曲线")
    plt.grid()
    plt.show()

if __name__=="__main__":
    train(x,y,coef)