import torch
import torch.nn
from torchvision import datasets,transforms
from torch.utils.data import random_split
'''
交叉验证 1.
# Train set用来让模型学习更新参数
# val set用来挑选某个时间戳上很好的模型参数
# test set用来检测性能
'''

#1. 随机划分数据集
batch_size = 100

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

#测试模型载入
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)
print(train_loader)
len1 = int(len(train_loader)/5 * 4)
len2 = int(len(train_loader)/5 )  
train_db,val_db = random_split(train_loader,[len1,len2])
#test_db = random_split(test_loader)

print('train:',len(train_db),'val:',len(val_db),'test:',len(test_loader))
