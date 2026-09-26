import numpy as np
import torch
#自动微分系统的实际运用
#1先正向传播算出预测值
#2基于损失函数 结合预测值和真实值 计算损失函数对预测值的梯度
#3更新参数

#数据集
x=torch.ones(2,5) #输入值 即特征
y=torch.zeros(2,3) #输出值 即标签   

#模型
w=torch.randn(5,3,requires_grad=True) #权重
b=torch.randn(3,requires_grad=True) #偏置 

#预测值
pred=x@w+b #或者是pred=torch.matmul(x,w)+b

#选择损失函数
criterion=torch.nn.MSELoss(pred,y)
loss=criterion(pred,y)

#自动求导计算损失函数对预测值的梯度
loss.backward()  
#计算损失函数对预测值的梯度
print(w.grad)
print(b.grad)

#更新参数                 #还有一种写法
with torch.no_grad():    #w.data=w.data-0.01*grad            
    w=w-0.01*w.grad      #b.data=b.data-0.01*b.grad
    b=b-0.01*b.grad      #这样就不用写with torch.no_grad()了
