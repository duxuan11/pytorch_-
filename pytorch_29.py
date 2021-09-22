import torch
from torch import optim
import torch.nn
from torch.nn import args
#动量法 pytorch 中添加动量 momentum = args.momentum
'''
optimizer = torch.optim.SGD(model.parameters(),args.lr,momentum=args.momentum,
weight_decay=agrs.weight_decay)
Adam都是内部设置好的
'''
#loss 平躺很长一段时间了
''' 
法1: 自衰减
 'min'代表希望减少
scheduler = ReduceLROnPlateau(optimizer,'min')

监听loss
for epoch in ,,....:
    scheduler.step(loss)

法2: 强制衰减
scheduler = StepLR(optimizer,step_size=30,gamma = 0.1)
for epoch in range(100):
    scheduler.step()
    train(..)
    .... 
'''
