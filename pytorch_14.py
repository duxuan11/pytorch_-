#单层感知机
import torch
from torch.nn import functional as F
x = torch.randn(1,10)
w = torch.randn(1,10,requires_grad=True)
o = torch.sigmoid(x@w.t())
print(o.shape) #torch.Size([1, 1])

loss = F.mse_loss(torch.ones(1,1),o)
print(loss.shape)
loss.backward()
print(w.grad)
#w.grad就是单层感知机每一个权值
#有了w梯度 可以根据 w = w - lr*w' 反复迭代得到一个最优的权值
#使得x*w -> y 误差越小


