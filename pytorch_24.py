import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms
'''
测试方法 2 测试集上计算准确度
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
            nn.LeakyReLU(inplace=True), #nn.LeakyReLU(inplace=True)
            nn.Linear(200,200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200,10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self,x):
        x = self.model(x)
        return x 
 
device = torch.device('cuda:0') 
net = MLP().to(device)  #初始化完成
#net.parameters() 可以避免直接在优化器里传参[w1,b1,w2,....]
optimizer = optim.SGD(net.parameters(),lr=learning_rate)
print("net.parameters():={}".format(net.parameters()))
criteon = nn.CrossEntropyLoss() #交叉熵

#训练部分
for epoch in range(epochs):
    for batch_idx, (data,target) in enumerate(train_loader):
        '''
        view函数:这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，那不确定的地方就可以写成-1
        '''
        #print(data.shape)
        data = data.view(-1,28*28)
        #print(data.shape)
        '''打印输出
        torch.Size([200, 1, 28, 28])
        torch.Size([200, 784])  因此这个data是200行，view里的-1改成200是不会报错的
        '''
        data,target = data.to(device),target.cuda()
        logits = net(data) #data进入到forward中(计算图)  
        loss = criteon(logits,target) #预测值与真实值之间的交叉熵
        ''''
        总得来说，这三个函数的作用是先将梯度归零optimizer.zero_grad()，
        然后反向传播计算得到每个参数的梯度值loss.backward()，
        最后通过梯度下降执行一步参数更新optimizer.step()
        '''
        #根据交叉熵的值进行优化
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\t loss: {:.6f}".format(
                epoch,(batch_idx+100) * len(data), len(train_loader.dataset),
                100.* (batch_idx+100) / len(train_loader),loss.item()
            ))

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
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f})%\n'.format(
        test_loss,correct,len(test_loader.dataset),
        100.* correct / len(test_loader.dataset)
    ))