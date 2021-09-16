# 演示交叉熵
'''
 Cross Entropy = ∑p*log2(1/p)
'''
import torch
from torch.nn import functional as F
a = torch.full([4],1/4) #每人概率相等，如果其中一个人获奖 信息量高
print(a)
p1 = -(a*torch.log2(a)).sum()
print(p1)

a = torch.tensor([0.1,0.1,0.1,0.7]) #0.7获奖概率大，信息量会比较小
p2 = -(a*torch.log2(a)).sum()
print(p2)

#可以看出p1的信息量比p2的更高
print('----------------------------------------------')

x = torch.randn(1,784)
w = torch.randn(10,784)
logits = x@w.t() #[1,10]
print(logits)
print("logits:",logits)
'''
softmax作用 让原本的输入通过映射，映射到(0,1)的值，这些值累和为1(满足概率的性质)
多用于多分类任务 可以让大的更大，小的更小
'''
pred = torch.softmax(logits,dim=1)
print("after softmax:",pred)

print('----------------------------------------------')

'''
pred_cross = softmax+log+nll
CrossEntropyLoss就是把Softmax–Log–NLLLoss合并成一步
'''
pred_log = torch.log(pred)
pred_nll = F.nll_loss(pred_log,torch.tensor([3]))
pred_cross = F.cross_entropy(logits,torch.tensor([3]))

print("pred_nll:",pred_nll)
print("pred_cross:",pred_cross)


