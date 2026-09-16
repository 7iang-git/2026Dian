import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 

#ANN案例的实现步骤
'''
1搭建数据集 
2搭建神经网络
3模型训练
4模型测试
'''

#1 搭建数据集Dataset Dataloader在后面训练和测试的时候再分别定义
def create_dataset():
    #todo1加载CSV  读取数据集
    data=pd.read_csv('./data/data.csv')
    #print(f'data.head(): {data.head()}')
    #print(f'data.shape: {data.shape}') 
    #todo2获取x特征列和y标签列
    x,y=data.iloc[:,:-1],data.iloc[:,-1]
    #print(f'x.head(): {x.head()},x.shape: {x.shape}')
    #print(f'y.head(): {y.head()},y.shape: {y.shape}')
    #todo3把特征列统一为float类型
    x=x.astype(np.float32)
    #todo4划分数据集为训练集和测试集
    #参1：特征矩阵x 参2：标签向量y 参3：测试集比例 参4：随机种子 参5：是否分层采样
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3,stratify=y)
    #todo5数据集转换
    train_dataset=TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset=TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    #todo6返回训练集和测试集                  这是输入特征数量  这是输出标签数量
    return train_dataset, test_dataset,x_train.shape[1],len(np.unique(y))
#2 搭建神经网络
class PhonePriceModel(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.linear1=nn.Linear(input_dim,128)
        self.linear2=nn.Linear(128,256)
        self.output=nn.Linear(256,output_dim)
    def forward(self,x):
        x=torch.relu(self.linear1(x)) 
        x=torch.relu(self.linear2(x))
        x=self.output(x)
        return x
#3 模型训练
def train(train_dataset,input_dim,output_dim):
    #创建数据加载器
    train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    #创建模型对象
    model=PhonePriceModel(input_dim,output_dim)
    #选择损失函数
    criterion=nn.CrossEntropyLoss()
    #选择优化器
    optimizer=optim.SGD(model.parameters(),lr=0.001)
    #多轮训练模型
    epochs=50
    for epoch in range(epochs):
        #定义变量 记录一轮所有批次的总损失和一轮包含的批次数量
        total_loss,batch_num=0.0,0
        start=time.time()
        for x, y in train_loader:
            #切换模型状态为训练状态
            model.train()
            #前向传播
            y_pred=model(x)
            #计算损失
            loss=criterion(y_pred,y)
            #梯度清零，反向传播
            optimizer.zero_grad()
            loss.backward()
            #更新参数
            optimizer.step()
            #记录损失
            total_loss+=loss.item() #累计每批次的(平均)损失 关键底层在损失函数cross entropy参数里面
            batch_num+=1
        #至此一轮训练结束 打印损失和时间
        print(f'epoch: {epoch}, loss: {total_loss/batch_num:.4f}, time: {time.time()-start:.2f}s')
    #保存模型
    #参1：模型参数 参2：模型保存路径
    torch.save(model.state_dict(), './model/phone.pth')
    #查看模型参数
    #print(f'\n\nmodel.state_dict(): {model.state_dict()}\n\n')
#4 模型测试
def evaluate(test_dataset,input_dim,output_dim):
    #创建数据加载器
    test_loader=DataLoader(test_dataset,batch_size=8,shuffle=False)
    #创建模型对象
    model=PhonePriceModel(input_dim,output_dim)
    #加载模型参数
    model.load_state_dict(torch.load('./model/phone.pth'))
    #切换模型状态为评估状态
    model.eval()
    #定义变量 记录预测正确的样本数量
    correct=0
    #从DataLoader中取出批次数据
    for x, y in test_loader:
        #前向传播
        y_pred=model(x)
        #由于搭建网络时输出层没用softmax激活函数 所以这里需要用argmax()获取预测类别
        y_pred=torch.argmax(y_pred,1)
        #统计预测正确的样本数量
        correct+=(y_pred==y).sum()
    #计算准确率
    accuracy=correct/len(test_dataset)
    print(f'accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    train_dataset, test_dataset,input_dim,output_dim=create_dataset()
    #训练时取消注释
    #train(train_dataset,input_dim,output_dim)
    #测试时取消注释
    evaluate(test_dataset,input_dim,output_dim)
