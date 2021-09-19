from pytorch_27 import MLP
from typing import NewType
import torch
import torch.optim as optim
'''
防止过拟合 
1. 正则化，消去高次项 weight decay
    L1-regularization λ∑|W|
    L2-regularization
'''
learning_rate = 1e-3
device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(),lr=learning_rate,weight_decay=0.01)

#L1-regularization λ∑|W|
'''
regularization_loss = 0
for param in model.parameters():
    regularization_loss += torch.abs(param)

classify_loss = criteon(logit,target)
loss = classify_loss + 0.01 * regularization_loss
optimizer.zero_grad()
loss.backward()
optimizer.step()
'''