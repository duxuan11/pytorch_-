import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms

'''
多分类问题模拟
模拟三层全连接层
'''

batch_size = 200
learning_rate = 1e-2
epochs = 10

#加载操作
'''
PyTorch中数据读取的一个重要接口是torch.utils.data.DataLoader，该接口定义在dataloader.py脚本中，只要是用PyTorch来训练
模型基本都会用到该接口，该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch size封装成Tensor，
后续只需要再包装成Variable即可作为模型的输入，因此该接口有点承上启下的作用，比较重要。这篇博客介绍该接口的源码，主要包含DataLoader
和DataLoaderIter两个类。
'''
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)



'''
        ch-out  ch-in
torch.randn(200,784,requires_grad=True) 类似卷积
torch.zeros(200,requires_grad=True) 类似图像堆叠
'''
w1,b1 = torch.randn(200,784,requires_grad=True),\
    torch.zeros(200,requires_grad=True) 
w2,b2 = torch.randn(200,200,requires_grad=True),\
    torch.zeros(200,requires_grad=True)
w3,b3 = torch.randn(10,200,requires_grad=True),\
    torch.zeros(10,requires_grad=True)


'''
对w1,w2,w3初始化
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)
'''


#前向传播
def forward(x):
    x = x@w1.t() + b1
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x@w3.t() + b3
    x = F.relu(x)
    return x

optimizer = torch.optim.SGD([w1,b1,w2,b2,w3,b3],lr=learning_rate)
criteon = torch.nn.CrossEntropyLoss()
for epoch in range(epochs):
    for batch_idx,(data,target) in enumerate(train_loader):
        data = data.view(-1,28*28)
        logits = forward(data)
        loss = criteon(logits,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\t loss: {:.6f}".format(
                epoch,batch_idx * len(data), len(train_loader.dataset),
                100.* batch_idx / len(train_loader),loss.item()
            ))
    
    test_loss = 0
    correct = 0
    for data,target in test_loader:
        data = data.view(-1,28*28)
        logits = forward(data)
        test_loss += criteon(logits,target).item()
        
        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f})%\n'.format(
        test_loss,correct,len(test_loader.dataset),
        100.* correct / len(test_loader.dataset)
    ))

'''
Train Epoch: 2 [20000/60000 (33%)]       loss: 2.302583
Train Epoch: 2 [40000/60000 (67%)]       loss: 2.302583
loss不变说明 你的梯度没变,在这里说明你的梯度为0
原因:
    1.learing——rate 过大
    2.梯度消失
    3.w初始化不对
'''