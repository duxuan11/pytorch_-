import torch
import  torch.nn.functional as F
x = torch.randn(1,784)
'''
layer = torch.nn.Linear(784,200)
x = layer(x)
这里就相当于 x@w.t()+b

因为又些元素是线性不可分的
我们可以设计一种神经网络，通过激活函数来使得这组数据线性可分。
激活函数我们选择阀值函数（threshold function），也就是大于某个
值输出（被激活了），小于等于则输出0（没有激活）。这个函数是非线性函数。

激活函数的另一个作用，特征的充分组合。
https://zhuanlan.zhihu.com/p/27661298
'''
layer1 = torch.nn.Linear(784,200)

layer2 = torch.nn.Linear(200,200)

layer3 = torch.nn.Linear(200,10)


x = layer1(x)
x =F.relu(x,inplace=True)
print("After layer1: x.shape = {}".format(x.shape))
x = layer2(x)
x =F.relu(x,inplace=True)
print("After layer2: x.shape = {}".format(x.shape))
x = layer3(x)
x =F.relu(x,inplace=True)
print("After layer3: x.shape = {}".format(x.shape))


'''
注意在pytorch中 autograd会自动记录向后求导的过程
自己要写好forward
'''