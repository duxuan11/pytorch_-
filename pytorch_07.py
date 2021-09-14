#broadingcast 注意boradcast使用情境
import numpy as np
import torch
#boradingcast就是根据需求，插入新的维度和扩张
#(unsqueence expand)
a = torch.randn(1,1,8)
b = torch.tensor(5.0)
b = b.unsqueeze(0).unsqueeze(0)
b = b.expand_as(a)
print(a)
print('\n')
print(a+b)
#match from last dim
