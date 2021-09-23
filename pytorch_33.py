import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

'''
Batch Norm 把输入的值控制在有效的范围内
如 -100 - 1000  -> 等效变换到 N（0，^）
'''
'''
Image Normalization 
mean 有三个值 表示 RGB 三个通道 【统计imagenet得来的】
使得RGB数据符合一个比较好的分布 
'''
normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std = [0.229,0.224,0.225])

'''
Batch Normalization 
->N(0,1) -> 继续学习β γ 最后符合
'''
x = torch.rand(100,16,784) #均值0.5方差是1
layer = nn.BatchNorm1d(16) #你有多少channel 我就统计几个channel上的均值
'''
 μ  σ²
'''
out = layer(x)
print(layer.running_mean) # μ
print(layer.running_var) # σ²


print("------------------------------------------------")

x = torch.rand(1,16,7,7)
layer = nn.BatchNorm2d(16)
out = layer(x)
print("layer.weight: {}".format(layer.weight))
print("layer.weight.shape: {}".format(layer.weight.shape))
print("layer.bias.shape: {}".format(layer.bias.shape))

print("------------------------------------------------")
# 在test过程中需要eval
print(layer.eval())
