import torch
a = torch.full([8],1)
b = a.reshape(2,4)
c = a.reshape(2,2,2)
print(a,b,c)
print('-------------------------------------------')
#1范式，所有元素绝对值求和
print(a.norm(1))
print(b.norm(1))
print(c.norm(1))
print('-------------------------------------------')
#2范式，所有元素绝对值平方再求和
print(a.norm(2))
print(b.norm(2))
print(c.norm(2))
print('-------------------------------------------')
print(b.norm(1,dim=1)) #所有行元素相加
print(b.norm(2,dim=1))
print(b.norm(1,dim=0)) #所有列元素相加
print('-------------------------------------------')
a = torch.arange(8).reshape(2,4).float()
#prod是累乘 argmax最大值索引[打平]
print(a.min(),a.max(),a.mean(),a.prod(),a.sum(),a.argmax(),a.argmin())
'''
如果我只是想在某个维度上求最大值和最小值或者其它，就要指定dim=
'''

print(a.min(dim=0),a.max(dim=1),a.argmax(dim=1))
print('-------------------------------------------')
a = torch.rand(1,10)
print(a)
print(a.max(dim=1)) #输出大小和指数
print(a.max(dim=1,keepdim=True)) #多加了一对括号
print('-------------------------------------------')
print(a.topk(3,dim=1)) #求第一维最大的3个
print(a.topk(3,dim=1,largest=False)) #求第一维最小的3个
print(a.kthvalue(3,dim=1)) #kthvalue只能是第k小的
print('-------------------------------------------')
print(a>0,torch.gt(a,0),torch.eq(a,a))