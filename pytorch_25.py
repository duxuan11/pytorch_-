
from typing import Optional
import numpy as np
import torch 
import torch.nn.functional as F
from torch.nn.modules import linear
import torch.optim as optim
from torch.tensor import Tensor
from torchvision import datasets, transforms
import torch.nn as nn

'''
visdom 可视化
visdom相当于一个web服务器，确保你运行程序之前就打开了visdom
python -m visdom.server
复制地址，打开浏览器
参考博客
https://blog.csdn.net/weixin_34910922/article/details/115920820
'''
from visdom import Visdom
#创建实例
viz = Visdom()
'''
创建一条直线
        x = 0 y = 0  ID = train_loss(它会去找有没有一个id为train_loss的窗口
        ，如果没有就创建一个，opts=dict 给定一些额外的信息) 
'''
def viz_test():
    viz.line([0.],[0.],win='train_loss',opts=dict(title = 'train loss'))
    '''
    在 win窗口里append loss,item数据
    '''
    viz.line([1,2,3,4,5,6],[1,2,3,4,5,6],win='train_loss',update='append')


    viz.line([1,5,6,8],[5,2,3,4],win='test',opts=dict(title = 'test loss&acc.',
    legend=['loss','acc.']))




batch_size = 200
learning_rate = 1e-2
epochs = 10

#训练模型载入
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


class MLP(nn.Module):


    def __init__(self):
        super(MLP,self).__init__()
       
        self.model =  nn.Sequential(
            nn.Linear(784,200),
            nn.LeakyReLU(inplace=True), #nn.LeakyReLU(inplace=True)
            nn.Linear(200,200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200,10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self,x):
        x = self.model(x)
        return x 
def module_start():
    device = torch.device('cuda:0') 
    net = MLP().to(device)  #初始化完成
    #net.parameters() 可以避免直接在优化器里传参[w1,b1,w2,....]
    optimizer = optim.SGD(net.parameters(),lr=learning_rate)
    print("net.parameters():={}".format(net.parameters()))
    criteon = nn.CrossEntropyLoss() #交叉熵
    viz.line([0.],[0.],win='train_loss',opts=dict(title = 'train loss',
            legend=['loss','step'],xtickmin=0,xtickmax = 200))
    viz.line([0.],[0.],win='Accuracy',opts=dict(title = 'Accuracy',
            legend = ['Accuracy','step']   
        ))
      
    #训练部分
    for epoch in range(epochs):
        for batch_idx, (data,target) in enumerate(train_loader):
        
            data = data.view(-1,28*28)
            data,target = data.to(device),target.cuda()
            logits = net(data) #data进入到forward中(计算图)  
            loss = criteon(logits,target) #预测值与真实值之间的交叉熵
            #print(loss.item())
            '''
            一定注意,y,x 只能是tensor 不能是张量 你得转[] 
            貌似 也就是[](list) 也是可以的
            '''
            viz.line(torch.tensor([loss.item()]),torch.tensor([batch_idx]),win='train_loss',update='append')
            #根据交叉熵的值进行优化
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            

        #测试精度，准确率    
        test_loss = 0
        correct = 0
        for data,target in test_loader:
            data = data.view(-1,28*28)
            data,target = data.to(device),target.cuda()
            logits = net(data)
            test_loss += criteon(logits,target).item()
            
            pred = logits.argmax(dim = 1)
            correct += torch.eq(pred,target).float().sum().item()

        test_loss /= len(test_loader.dataset)
        viz.line(torch.tensor([100.* correct / len(test_loader.dataset)]),torch.tensor([epoch]),
         win='Accuracy',update='append')
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f})%\n'.format(
            test_loss,correct,len(test_loader.dataset),
            100.* correct / len(test_loader.dataset)
        ))

if __name__=='__main__':
    module_start()
    