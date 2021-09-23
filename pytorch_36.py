from _typeshed import Self
from torch import optim
import torch.nn as nn
import torch
from torch.optim import optimizer 

'''
nn.modules 出现类的嵌套
'''
class BasicNet(nn.Module):
    def __init__(self) -> None:
        #super规范 当前类名，self
        super(BasicNet,self).__init__()
        self.net = nn.Linear(4,3)
    
    def forward(self,x):
        return self.net(x)

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net,self).__init__()
        self.net = nn.Sequential(BasicNet(),
                                nn.ReLU(),
                                nn.Linear(3,2))
    def forward(self,x):
        return self.net(x)
device = torch.device('cuda')
net = Net()
net2 = net
net.to(device)  #用不用Gpu 对nn.module不改变无影响 都是同一个 
#而tensor是不一样的
'''
t1 = torch.tensor([1,2])
t2 = t1.to(device)
print("对t1使用GPU之后得到的t2与t1对比{}".format(t1 == t2))
'''

#print(dict(net.named_parameters()).items())
# for name, t in net.named_parameters():
#         print('parameters:', name, t.shape)

# for name, m in net.named_children():
#         print('children:', name, m)


# for name, m in net.named_modules():
#         print('modules:', name, m)

'''
parameters: net.0.net.weight torch.Size([3, 4])    basicnet 
parameters: net.0.net.bias torch.Size([3])
parameters: net.2.weight torch.Size([2, 3])        linear
parameters: net.2.bias torch.Size([2])
children: net Sequential(
  (0): BasicNet(
    (net): Linear(in_features=4, out_features=3, bias=True)
  )
  (1): ReLU()
  (2): Linear(in_features=3, out_features=2, bias=True)
)
#它本身
modules:  Net(
  (net): Sequential(
    (0): BasicNet(
      (net): Linear(in_features=4, out_features=3, bias=True)
    )
    (1): ReLU()
    (2): Linear(in_features=3, out_features=2, bias=True)
  )
)
#直系孩子
modules: net Sequential(
  (0): BasicNet(
    (net): Linear(in_features=4, out_features=3, bias=True)
  )
  (1): ReLU()
  (2): Linear(in_features=3, out_features=2, bias=True)
)
#孙子
modules: net.0 BasicNet(
  (net): Linear(in_features=4, out_features=3, bias=True)
)
modules: net.0.net Linear(in_features=4, out_features=3, bias=True)

#孙子
modules: net.1 ReLU()
modules: net.2 Linear(in_features=3, out_features=2, bias=True)
'''

'''
保存和加载类文件

1.加载模型 load_state_dict将网络模型里面的参数都初始化为train好的值
net.load_state_dict(torch.load('ckpt.mdl'))

#train.. 训练时候保存，防止突然结束,什么都没有了
torch.save(net.state_dict(),'ckpt.mdl')
'''

'''
net.train() 可以将所有子节点都变成train状态
net.eval() 可以将所有子节点都变成test状态
'''

class Flatten(nn.Module):
    def __init__(self) -> None:
        super(Flatten,self).__init__()
    
    def forward(self,input):
        return input.view(input.size(0),-1)

'''
为什么要创建一个flatten类 因为从卷积层到全连接层需要打平【也就是要reshape】可以把他放到flatten
不用写俩遍Sequential中间加上view操作了
因为只有类才可以写到Sequential里面去
'''
class TestNet(nn.Module):
    def __init__(self) -> None:
        super(TestNet,Self).__init__()
        self.net = nn.Sequential(nn.Conv2d(1,16,stride=1,padding=1),
                                nn.MaxPool2d(2,2),
                                Flatten(),
                                nn.Linear(1*14*14,10))
    
    def forward(self,x):
        return self.net()



class MyLinear(nn.Module):
    
    def __init__(self,inp,outp) -> None:
        super(MyLinear,self).__init__()
        #将该tensor变量加到nn.Parameter类里，他就会被加入到
        #net.parameters() 然后被SGD优化  听且net.parameters()方法会加上这些tensor
        self.w = nn.Parameter(torch.randn(outp,inp)) #w 维度是 输出*输入
        self.b = nn.Parameter(torch.randn(outp))
    
    def forward(self,x):
        x = x @ self.w.t() + self.b
        return x