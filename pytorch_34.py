import torch.nn as nn
import torch
import torch.nn.functional as F
'''
1*1 conv 怎么理解
原来 w*h*6 -> 1*1*5 相当于左边6个特征图 每个都全连接1*1卷积核 最后输出5个特征图 
concat 就是特征图堆叠
6*7*7 + 12*7*7 = 18*7*7
在计算参数的过程中:
注意参数 只是涉及权值和偏执，跟图像大小无关
ResNet
不加shortcut 你可能是个非常曲折的面，有很多局部最小值
加了shortcut 你可能就是一个平滑的曲面，没有那么多的局部最小值
'''
class ResBlk(nn.Module):
    def __init__(self,ch_in,ch_out) -> None:
        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn1   = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1)
        self.bn2   = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1),      
            ) 
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.extra(x) + out
        return out
