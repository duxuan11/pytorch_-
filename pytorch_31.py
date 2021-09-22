import torch
import torch.nn as nn
import torch.nn.functional as F
'''
卷积 有多少条线就有多少参数量
如: 4 Hidden Layers: [784,256,256,256,256,10]
因此总参数量: 784*256 + 256*256 + 256*256+ 256*256 + 256*10 = 399,872
'''

'''
二维卷积神经网络
'''
'''
input channel kernal num 卷积核的大小 步长 填充   
输入的通道数 输出的通道数 卷积核的大小 
'''
'''
输入图片大小 W×W
Filter大小 F×F
步长 S
padding的像素数 P
N = (W − F + 2P )/S + 1
'''
layer = nn.Conv2d(3,6,kernel_size=3,stride=1,padding=0)
#x表示 5张图片 每张3个通道 大小28*28
x = torch.rand(5,3,28,28)
print("输入x图像的信息：{}".format(x.shape))

out = layer.forward(x)
#输出是指5张图片 6个通道(6个卷积特征核)
print("输出x图像的信息：{}".format(out.shape))

print("------------------------------------------------")
layer1 = nn.Conv2d(3,3,kernel_size=3,stride=1,padding=1)
x1 = torch.rand(1,3,28,28)
print("输入x1图像的信息：{}".format(x1.shape))
out1 = layer1.forward(x1)
print("输出x1图像的信息：{}".format(out1.shape))

print("------------------------------------------------")
layer2 = nn.Conv2d(1,3,kernel_size=3,stride=2,padding=0)
x2 = torch.rand(1,1,28,28)
print("输入x2图像的信息：{}".format(x2.shape))
out2 = layer2(x2)
print("输出x2图像的信息：{}".format(out2.shape))

#注意，我们要使用layer(x)来代替layer.forward(x)
#我们调用类的实例(layer) pytroch会先调用hooks然后才会调用forward 要比直接调用forward要好

print("------------------------------------------------")
'''
权重
'''
#print("layers的w权重：{}".format(layer2.weight))
'''
requires_grad=True 说明 w与b是会自动更新的
bias 跟输出通道数保持一致
'''
print("layers的大小{},bias的大小{}".format(layer2.weight.shape,layer2.bias.shape))
print("------------------------------------------------")
'''
w [卷积核的个数，输入通道数，图像尺寸]
'''
w = torch.rand(16,3,5,5)
b = torch.rand(16) #卷积核个数对应
out = F.conv2d(x,w,b,stride=1,padding=1)
'''
最后计算应该得到out 为[5,16,26,26]
'''
print(out.shape)

w = torch.rand(6,3,3,3)
b = torch.rand(6) #卷积核个数对应
out = F.conv2d(x,w,b,stride=2,padding=2)
'''
最后计算应该得到out 为[5,6,15,15]
'''
print(out.shape)