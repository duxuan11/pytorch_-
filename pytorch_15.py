import torch
from torch import autograd
from torch.nn import functional as F
x = torch.randn(1,10)
w = torch.randn(2,10,requires_grad=True)
o = torch.sigmoid(x@w.t())
print(o.shape)

loss = F.mse_loss(torch.ones(1,1),o) #广播机制
print(loss)
loss.backward()
print(w.grad) # w.grad肯定和w一样的shape 2*10就是k=[0,1] w = [0,9] 9->2
print('----------------------------------------------')
'''
链式法则
'''
x = torch.tensor(1.)
w1 = torch.tensor(2.,requires_grad=True)
b1 = torch.tensor(1.)
w2 = torch.tensor(2.,requires_grad=True)
b2 = torch.tensor(1.)

y1 = x*w1+b1
y2 = y1*w2+b2

dy2_dy1 = autograd.grad(y2,[y1],retain_graph=True)[0]
dy1_dw1 = autograd.grad(y1,[w1],retain_graph=True)[0]
dy2_dw1 = autograd.grad(y2,[w1],retain_graph=True)[0]
'''
dy2/dw1 = dy2/dy1 * dy1/dw1 
'''
print(dy2_dy1*dy2_dw1)
print(dy2_dw1)