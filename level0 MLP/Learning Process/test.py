#黑马程序员学习笔记
import torch

#参数1 数据原始值 参数2 是否自动微分  参数3 数据类型
w=torch.tensor(10,requires_grad=True,dtype=torch.float32)

#损失函数
loss=w**2+20

#梯度下降法
for i in range(1,101):
    #1 正向传播
    loss=w**2+20
    
    #2 梯度清零 因为梯度默认会进行累加  第一次因为没有反向传播 w.grad是none会报错
    if w.grad is not None:
        w.grad.zero_() 
    
    #3 反向传播 得到梯度  梯度等于loss函数的导数
    loss.sum().backward()
    
    #4梯度更新
    w.data=w.data-0.01*w.grad
    print(f"第{i}次的(0.01*w.grad )为:{0.01*w.grad}")