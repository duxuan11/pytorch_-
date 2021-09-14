import torch
a = torch.linspace(-100,100,10)
print(a)
print(torch.sigmoid(a))
print('-------------------------------------------')
a = torch.linspace(-1,1,10)
print(torch.tanh(a))
print('-------------------------------------------')
'''
使用梯度函数的俩种方式
'''
from torch.nn import functional as F
a = torch.linspace(-1,1,10)
print(torch.relu(a))
print(F.relu(a))
print('-------------------------------------------')
'''
pred = x * w + b [1,[2],0]
'''
x = torch.ones(1)
w = torch.full([1],2)
mse = F.mse_loss(torch.ones(1),x*w) #输入值，目标值 （predict label）
#tensor(1.)
print(mse)
'''
print(torch.autograd.grad(mse,[w])) #loss ,[自变量w1,w2,w3,w4]
发生异常: RuntimeError
element 0 of tensors does not require grad and does not have a grad_fn
'''
w.requires_grad_()
#也可以直接创建 w = torch.full([1],2,requires_grad=True)
'''
print(torch.autograd.grad(mse,[w]))
发生异常: RuntimeError
element 0 of tensors does not require grad and does not have a gra
因为pytorch是动态建图的过程，之前写的mse里w是不能计算梯度的
'''
'''

'''
mse = F.mse_loss(torch.ones(1),x*w)
print(torch.autograd.grad(mse,[w]))
mse.backward() #在loss调用backward,自动的往后传播(pytorch会自动建图)
#会把loss相对于任意输入的梯度，计算出来放在变量名.grad里
print(w.grad)