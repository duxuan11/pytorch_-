from torch import optim
import torch.nn as nn
import torch
from torch.optim import optimizer 
'''
nn.Module
如果要实现自己的类的话，要继承自nn.Module类
在初始化里面完成自己的逻辑，在forward里面完成自己的前向构建的过程
*** 1.继承nn.Module的好处 
      可以使用现有的函数 nn.Linear nn.BatcHNorm2d nn.Conv2d  nn.ReLU nn.Sigmoid nn.ConvTransposed2d(转置卷积层) nn.Dropout            
    2.Container
      self.net = nn.Sequential()  self.net自动完成 forward过程
'''
class MyLinear(nn.Module):
    def __init__(self,inp,outp) -> None:
        super(MyLinear,self).__init__()
        self.w = nn.Parameter(torch.randn(outp,inp))
        self.b = nn.Parameter(torch.randn(outp))

    def forward(self,x):
        x = x @ self.w.t() + self.b
        return x

'''
nn.Linear()为全连接层
 PyTorch的nn.Linear（）是用于设置网络中的全连接层的，需要注意在二维图像处理的
任务中，全连接层的输入与输出一般都设置为二维张量，形状通常为[batch_size, size]，
不同于卷积层要求输入输出是四维张量。
(class) Linear(in_features: int, out_features: int, bias: bool = ...)
in_features指的是输入的二维张量的大小，即输入的[batch_size, size]中的size。
out_features指的是输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，
当然，它也代表了该全连接层的神经元个数。
y = x Wt + b 
因此wt的形状应该为 [2,4]
'''
net = nn.Sequential(nn.Linear(4,2),nn.Linear(2,2))

print(list(net.parameters())[0].shape)
#torch.Size([2, 4])
print(list(net.parameters())[3].shape)
print(list(net.named_parameters())[0])
'''  
('0.weight', Parameter containing:
tensor([[ 0.3517, -0.1755, -0.4091, -0.0041],
        [ 0.3975, -0.3571, -0.2364,  0.2689]], requires_grad=True))
'''
print(list(net.named_parameters())[1])
print(dict(net.named_parameters()).items())
'''
dict_items([('0.weight', Parameter containing:
tensor([[-0.2546, -0.2176, -0.3504,  0.1527],
        [-0.4797,  0.1279, -0.4687, -0.1287]], requires_grad=True)), ('0.bias', Parameter containing:
tensor([ 0.2098, -0.0238], requires_grad=True)), ('1.weight', Parameter containing:    
tensor([[-0.0273, -0.4071],
        [-0.1502,  0.5691]], requires_grad=True)), ('1.bias', Parameter containing:    
tensor([-0.4891, -0.4847], requires_grad=True))])
'''

#这就可以解释，当使用优化器时候，net.parameters就能够把所有层的权值和偏置传递进去了
optimizer = optim.SGD(net.parameters(),lr=1e-3)