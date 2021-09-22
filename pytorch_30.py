import torch 
import torch.nn as nn

'''
1. Early Stopping 

2. Dropout

3. Stochastic Gradient Descent
'''
'''
torch.nn.Dropout(p=dropout_prob) p = 1 全丢完 
tf.nn.dropout(keep_prob) p = 1  tensorflow 全部保持住
'''
class MLP(nn.Module):


    def __init__(self):
        super(MLP,self).__init__()
       
        self.model =  nn.Sequential(
            nn.Linear(784,200),
            nn.dropout(0.5), #drop 50% of the neuron
            nn.LeakyReLU(inplace=True), #nn.LeakyReLU(inplace=True)
            nn.Linear(200,200),
            nn.dropout(0.5), #drop 50% of the neuron
            nn.LeakyReLU(inplace=True),  
            nn.Linear(200,10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self,x):
        x = self.model(x)
        return x 

'''
注意: train的时候可以用dropout 但是val的时候就不要用 
for epoch in range(epochs):
    net_dropped.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        net_dropped.eval()**** 封住dropout()
'''

'''
Stochastic x -> f(x) ~ N(0,x) 
把原来在所有图片上的梯度之和变为到，取1个batch-size大小的样本的梯度之和
batch = 8,16,32,64...
Deterministric x -> f(x) 
'''
