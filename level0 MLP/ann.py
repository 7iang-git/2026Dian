import torch
import torch.nn as nn
from torchsummary import summary    #用于查看神经网络的参数信息

#1 搭建神经网络
class My_Model(nn.Module):
    def __init__(self):
        #第一步：继承nn.Module
        super().__init__()
        #第二步：搭建隐藏层和输出层
        self.linear1 = nn.Linear(3, 3)
        self.linear2 = nn.Linear(3, 2)
        self.output = nn.Linear(2, 2)
        
        #第三步：参数初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.kaiming_normal_(self.output.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
        nn.init.zeros_(self.output.bias)
        #第四步：forward方法
    def forward(self, x):
        x= self.linear1(x)
        x=torch.sigmoid(x)  #加权求和+激活函数  分解版写法
        x=torch.relu(self.linear2(x))  #合并版写法
        x=torch.softmax(self.output(x), dim=1)
        return x

#2 模型训练
def train():
    #创建模型对象
    my_model = My_Model()  
    #创建样本
    data= torch.randn(5, 3)
    #调用神经网络->进行训练
    output = my_model(data)  #这句话自动调用了forward方法
    print(f'output: {output}')
    #查看神经网络的参数信息
    print('===========神经网络的参数信息===========')
    summary(my_model, input_size=(5,3))
    for name, param in my_model.named_parameters():
        print(f'name: {name}, param: {param}\n')

if __name__ == '__main__':
    train()