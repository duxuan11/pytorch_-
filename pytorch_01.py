import torch 
import time 
print(torch.__version__)
print(torch.cuda.is_available())
k = torch.randn(10,1)
print(k)
'''
#标准正态分布 N(0,1)均值为0，方差为1 torch.randn(10,1)
#均匀分布 torch.rand()  [0,1)
俩个api内的参数表示二维数据m*n 
'''
a = torch.randn(10000,1000)
b = torch.randn(1000,2000)

#cpu模式的矩阵乘法
t0 = time.time()
c = torch.matmul(a,b)
t1 = time.time()
print(a.device,t1-t0,c.norm(2))

#gpu模式的矩阵乘法
device = torch.device('cuda')
a = a.to(device)
b = b.to(device)

#GPU第一次运算初始化，会消耗很多时间，但是之后会运算很快
t0 = time.time()
c = torch.matmul(a,b)
t1 = time.time()
print(a.device,t1-t0,c.norm(2))#c.norm(2)求矩阵或向量的范数

t0 = time.time()
c = torch.matmul(a,b)
t1 = time.time()
print(a.device,t1-t0,c.norm(2))

# -------------------------------------------------------------------
# 输出
# 1.2.0
# True
# cpu 0.24418282508850098 tensor(140959.5156)
# cuda:0 0.2847774028778076 tensor(141358.6094, device='cuda:0')
# cuda:0 0.0019979476928710938 tensor(141358.6094, device='cuda:0')
# -------------------------------------------------------------------

#深度学习就是导数的编程
import torch 
from torch import autograd
#在pytorch中，张量tensor是最基本的运算单位，
#与Numpy中的NDArray类似，张量表示一个多维矩阵
#但tensor可以在gpu运行但NDArray只能在cpu上运行
x = torch.tensor(1.)
a = torch.tensor(1.,requires_grad=True)
b = torch.tensor(2.,requires_grad=True)
c = torch.tensor(3.,requires_grad=True)

y = a**2 * x + b * x + c
print('before:',a.grad,b.grad,c.grad)

#y对abc求梯度,偏微分
grads = autograd.grad(y,[a,b,c])
print('after:',grads[0],grads[1],grads[2])
