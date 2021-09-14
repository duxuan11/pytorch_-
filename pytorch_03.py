import torch
import numpy as np

#标量和向量

a = torch.randn(2,3) #正态分布N(0,1) 二行三列的数据，默认是floatTensor
#判断元素类型
print(a.type(),type(a),torch.Tensor,isinstance(a,torch.FloatTensor))
#tensor
#loss常常用标量表示 一个标量就是一个单独的数 0维 len(a.shape)
print(torch.tensor(1.),torch.tensor(1.3))
b = torch.tensor(2.2) #维度为0的标量
print(b.shape,len(b.shape),b.size())

#向量
vector_a = torch.tensor([1.1]) #di为1的向量 且指定数据
vector_a_float = torch.FloatTensor(1) #随机初始化1个1维数据 

#one不带[]就是标量 shape表示其size
data = np.ones(2) # 1维2个1
torch_data_ones = torch.ones(2)
torch_data = torch.from_numpy(data)
print(vector_a,vector_a_float,data,torch_data,torch_data_ones.shape)

#利用shape和size索引
c = torch.randn(2,3)
#c.shape[1]  表示这一行有几列
# #c.size(0) 表示这一列有几行
print(c.size,c.shape,c.shape[1],c.size(0))

#1个2维三列的三维数组
d = torch.rand(1,2,3)
print(d,d.shape,d[0],list(d.shape)) #list可以把tensor数组转成List

#图片适合四维数据
e = torch.rand(2,3,28,28) #[B,C,H,W]
print(e,e.shape)

#其它常用函数
#num of element
print(e.numel())
#维度
print(e.dim())


#错误torch.tensor(3,4) 只接收单个数，或者指定好[]
#注意 torch.tensor() 小写的tensor接受的是数据 Tensor() FloatTensor接收的是类型和带[]具体数据
#以后大写的都给shape，小写的用来创建标量和具体数据[]