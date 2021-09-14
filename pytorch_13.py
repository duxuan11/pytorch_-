import torch
from torch.nn import functional as F
a = torch.rand(3,requires_grad=True)
p = torch.softmax(a,dim=0)
'''
发生异常: RuntimeError 
p.backward() 这里的p是向量
grad can be implicitly created only for scalar outputs #标量输出 backward只能为标量输出隐式创建梯度
'''
print(a)
print(p)
print('----------------------------------------------')
print(torch.autograd.grad(p[1],[a],retain_graph=True))
print(torch.autograd.grad(p[2],[a]))
'''
(tensor([-0.0925,  0.2195, -0.1270]),) p对a求偏导，相同大于0，不同小于0
(tensor([-0.1110, -0.1270,  0.2379]),)
'''

