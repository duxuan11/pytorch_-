from pytorch_21 import MLP
import torch 
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn as nn
#使用GPU加速
'''
注意，tensor.cuda 会返回一个新对象，这个新对象的数据已转移至GPU，而之前的 tensor 还在原来的设备上
（CPU ）。而 module.cuda 则会将所有的数据都迁移至 GPU ，并返回自己。所以 module = module.cuda() 
和 module.cuda() 所起的作用一致。
重新赋给自己，tensor 指向 cuda 上的数据，不再执行原数据。不指定使用的 GPU 设备，将默认使用第 1 块 GPU。
'''
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
            nn.ReLU(inplace=True), #nn.LeakyReLU(inplace=True)
            nn.Linear(200,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,10),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.model(x)
        return x 
 
 
device = torch.device('cuda:0') #后面数字表示显卡号
net = MLP().to(device) #模型.to() 加载显卡
optimizer = optim.SGD(net.parameters(),lr=learning_rate)
criteon = nn.CrossEntropyLoss() #交叉熵


for epoch in range(epochs):
    for batch_idx, (data,target) in enumerate(train_loader):
        data = data.view(-1,28*28)
        data,target = data.to(device), data.cuda() #
        logits = net(data)
        loss = criteon(logits,target)        
        #根据交叉熵的值进行优化
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\t loss: {:.6f}".format(
                epoch,batch_idx * len(data), len(train_loader.dataset),
                100.* batch_idx / len(train_loader),loss.item()
            ))
#测试部分    
test_loss = 0
correct = 0
for data,target in test_loader:
    data = data.view(-1,28*28)
    data,target = data.to(device), data.cuda() #
    logits = net.forward(data)
    test_loss += criteon(logits,target).item()
    
    pred = logits.data.max(1)[1]
    correct += pred.eq(target.data).sum()

test_loss /= len(test_loader.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f})%\n'.format(
    test_loss,correct,len(test_loader.dataset),
    100.* correct / len(test_loader.dataset)
))