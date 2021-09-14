import torch
#拼接与拆分
#Merge or split
'''
cat stack split chunk
'''
#拼接俩个tensor
a = torch.rand(4,3,32,8)
b = torch.rand(5,3,32,8)
#俩个拼接，必须让其它元素都保持一样 [只有1个维度不一样]
print(torch.cat([a,b],dim=0).shape)
print("------------------------------------------")
c = torch.rand(5,3,32,8)
#创建一个新维度拼接
print(torch.stack([b,c],dim=2).shape) #torch.Size([5, 3, 2, 32, 8])
print("------------------------------------------")
#split根据长度拆分
e,f = a.split([2,2],dim=0) #split可以用[]分割
print(e.shape,f.shape) #torch.Size([2, 3, 32, 8]) torch.Size([2, 3, 32, 8])
e,f = a.chunk(2,dim=0) #chunk只能按大小2，固定
print(e.shape,f.shape) 
