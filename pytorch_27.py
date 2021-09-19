from os import name
import torch
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import optimizer
from torchvision import datasets,transforms
from torch.utils.data import random_split,DataLoader
import torch.optim as optim
import visdom 
from torch.nn import Linear 
import torch.tensor
'''
k-fold cross validation
k折交叉验证
merge train/val sets
randomly sample 1/k as val set
'''
k = 5
train_len = 60000
lr_rate = 1e-3
epochs = 10
vis = visdom.Visdom()

#1.数据集载入


batch_size = 100

#训练集
train_db = datasets.MNIST('../data',train=True,download=False,
transform=transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))]))


    
#测试集
test_db = datasets.MNIST('../data',train=True,download=False,
transform=transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.1307,), (0.3081,))]))

#训练集上拆分成训练集与测试集
train_db,val_db = random_split(train_db,[50000,10000])
    

trainloader = DataLoader(train_db,batch_size=batch_size,shuffle=True)
valloader = DataLoader(val_db,batch_size=batch_size,shuffle=True)
testloader = DataLoader(test_db,batch_size=batch_size, shuffle=True)


class MLP(nn.Module):
    #初始化
    def __init__(self) ->None:
        super(MLP,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784,200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200,200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200,10),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x = self.model(x)
        return x 

def train_val():
    '''
    1.加载模型 优化器
    '''
    device = torch.device('cuda:0')
    net = MLP().to(device)
    '''
    优化器,给参数和学习率
    '''
    optimizer = optim.SGD(net.parameters(),lr=lr_rate)
    '''
    使用交叉熵作为优化标准
    '''
    criteon = nn.CrossEntropyLoss() 

    for epoch in range(epochs):
        for batch_idx, (data,target) in enumerate(trainloader):
               
                data = data.view(-1,784)
                data,target = data.to(device),target.cuda()
                #MLP操作
                pred = net(data)
                loss = criteon(pred,target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                vis.line([loss.item()],[batch_idx],name='train_loss',
                win='train_val_loss',update='append')

                if batch_idx % 100 == 0:
                    print("Train Epoch: {} [{}/{} ({:.0f}%)]\t loss: {:.6f}".format(
                    epoch,(batch_idx+100) * len(data), len(trainloader.dataset),
                    100.* (batch_idx+100) / len(trainloader),loss.item()
                    ))
       


        for batch_idx, (data,target) in enumerate(valloader):
            data = data.view(-1,28*28)
            data,target = data.to(device),target.cuda()
            logits = net(data)
            val_loss = criteon(logits,target).item()
            vis.line([[val_loss]],[batch_idx],name='val_loss',
                win='train_val_loss',update='append')    

if __name__ == '__main__':

    '''
    加载VISDOM, 设置曲线属性
    ''' 
    opt1 = {
    "title": 'train_val_loss',
    "xlabel":'batch_size',
    "ylabel":'loss',
    "width":400,
    "height":400,
    "legend":['train_loss','val_loss']
    }
    #初始化线段
    vis.line([[0.0, 0.0]],[0.],win='train_val_loss',opts=opt1)
    vis.line([0.],[0.],win='Accuracy',opts=dict(title = 'Accuracy',
            legend = ['Accuracy','epoch']   
        ))

    train_val()