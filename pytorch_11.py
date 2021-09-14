#pytorch高阶操作
'''
    where(cond,A,B) GPU完成比FOR高效，高度并行
'''
import torch
#例子1
a = torch.tensor([1,2,3,9])
b = torch.tensor([6,7,8,1])
c = torch.where(a>5,a,b)
print(c)
print('-------------------------------------------')
#例子2
cond = torch.rand(2,2)
a = torch.zeros(2,2)
b = torch.ones(2,2)
c = torch.where(cond>0.5,a,b)
print(c)
print('-------------------------------------------')
'''
gather 是一个查表的过程
torch.gather(input表,dim=,[索引列表])
'''
prob = torch.randn(4,10)
idx = prob.topk(dim=1,k=3)
idx = idx[1] #获取索引
label = torch.arange(10)+100
#它将标签扩展到4,10 利用index 来控制维度大小和对应得指标
#relative - > global 
print(torch.gather(label.expand(4,10),dim=1,index=idx.long()))



