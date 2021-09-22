import torch.nn as nn
import torch
import torch.nn.functional as F
'''
池化与采样 只会改变长和宽

注意******: 下采样和池化应该是包含关系,池化属于下采样,而下采样不局限于池化,如果卷积 stride=2,此时也可以把这种卷积叫做下采样.
'''



layer = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=0)
#x表示 5张图片 每张3个通道 大小28*28
x = torch.rand(1,1,28,28)
print("输入x图像的信息：{}".format(x.shape))
#卷积
out1 = layer(x)
layer2 = nn.MaxPool2d(2,stride=2)
out2 = layer2(out1)
'''
结果 1*16*13*13
'''
print("第一次池化后的结果{}".format(out2.shape))

print("------------------------------------------------")
layer3 = nn.AvgPool2d(2,stride=2)
out3 = layer3(out2)
'''
不够池化的行与列直接丢掉 13*13 -> 6*6
'''
print("第二次池化后的结果{}".format(out3.shape))


print("------------------------------------------------")
#上采样 插值  【输入，放大尺寸，模式】 
out4 = F.interpolate(out3,scale_factor=2,mode='nearest')
'''
mode = nearest , linear , bilinear , bicubic , trilinear
'''
print('上采样之后的结果:{}'.format(out4.shape))
print("------------------------------------------------")
#Relu
layer4 = nn.ReLU(inplace=True) # x -> x' 大小尺寸没变，写上true可以重新存储在原来x的内容里,不用新创建内存 节省内存空间
'''
mode = nearest , linear , bilinear , bicubic , trilinear
'''
out5 = layer4(out2)
print('ReLu之后的结果:{}'.format(out5.shape))
print(out5)
