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
scheduler = ReduceLROnPlateau(optimizer,'min')
'''