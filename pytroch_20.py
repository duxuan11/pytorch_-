from pytorch_18 import forward
import torch 
import torch.nn as nn
import torch.nn.functional as F
#创建自己的深度学习网络层

class MLP(nn.Module):
    '''
    这个网络参数已经写好了784,200..因此不需要传参
    你要自己改的话需要传参
    
        def __init__(self):
        super(MLP,self).__init__()
    是nn.Module固定写法
    '''
    def __init__(self):
        super(MLP,self).__init__()
        '''
        nn.Sequential
        一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加
        到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传
        入参数。
        
        简言而知，将()里的，按顺序加入到计算图中
        '''
        self.model =  nn.Sequential(
            nn.Linear(784,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,10),
            nn.ReLU(inplace=True),
        )
    '''
    由于该类继承自nn.model 因此,传入参数x.
    它自动进入上方Sequential里的计算图
    '''
    def forward(self,x):
        x = self.model(x)
        return x 
        
'''
torch.nn的实现去调用torch.nn.functional，实现方式是一致的。它们的区别是：
nn可以写在深度学习模型的初始化中，其是一个类；F函数不可以，它是一个实际的函数，
其需要输入实际的input
nn.大写的
F.小写的
'''


x = torch.randn(1,10)
print(F.relu(x,inplace=True))
x = nn.ReLU(x)
print(x)