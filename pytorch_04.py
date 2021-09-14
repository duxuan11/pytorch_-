#tensor的创建
import numpy as np
import torch

#大数据量可以从numpy转过来，小数据量就没必要

a = np.array([2,3.4])
a_tro = torch.from_numpy(a)

b = np.ones([2,3])
b_tro = torch.from_numpy(b)

print(a_tro,b_tro)
print("------------------------------------------------------")
#未初始化的二行三列数据
print(torch.empty(2,3),torch.Tensor(2,3),torch.IntTensor(2,3),
torch.FloatTensor(2,3))
print("------------------------------------------------------")
print(torch.tensor([1.2,3]).type()) #默认类型是floattensor
#torch数据类型设置更大
torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1.2,3]).type())
print("------------------------------------------------------")
#随机初始化
print(torch.rand(3,3))
print("\n")
c = torch.rand(3,3)
d = torch.rand_like(c) #读出c的shape喂给d
f = torch.randint(1,100,[3,3]) #1-100之间 3*3数据
print(c)
print("\n")
print(d)
print("\n")
print(f)
print("\n")
print("------------------------------------------------------")
#正太分布初始化
e = torch.randn(3,3) #均值为0，方差为1的正太分布
#自定义正太分布
#mean=torch.full([10],0)生成长度为10且都为0的向量 
#std=torch.arange(1,0,-0.1) 方差从1到0逐渐减小
f = torch.normal(mean=torch.full([10],0),std=torch.arange(1,0,-0.1))
print(f)
print("------------------------------------------------------")
#tensor全部清0，赋值为同一元素
print(torch.full([2,3],1))
print(torch.full([],7)) #这里指的生成1个标量
print(torch.full([1],7)) #这里指的生成1个向量
print("------------------------------------------------------")
#遍历,设置等差数列
torch.arange(0,10)
torch.arange(0,10,2)
#等分,0-10等切4块
torch.linspace(0,10,steps=4)
#log切分
torch.logspace(0,-1,steps=10)
print("------------------------------------------------------")
#生成全1
torch.ones(3,3)
#生成全0
torch.zeros(3,3)
#生成对角线为1,单位阵只能接收1，2个参数
torch.eye(3,3)
print("------------------------------------------------------")
#随机打散,左闭右开
torch.randperm(10)
perm_1 = torch.randint(0,10,[2,3])
perm_2 = torch.randint(0,10,[2,2])
idx = torch.randperm(5) #shuffle
print(idx) #tensor([2, 0, 3, 4, 1])

print('\n')
print(perm_1[0],perm_2[0]) #tensor([5, 5, 1]) tensor([5, 2])

