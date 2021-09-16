'''
2D函数优化案例
f(x,y) = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
'''
import torch
from mpl_toolkits.mplot3d import Axes3D
from torch.nn import functional as F
import numpy as np
import matplotlib.pylab as plt
def himmelblau(x):
    return (x[0]**2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 -7)** 2

x = np.arange(-6,6,0.1)
y = np.arange(-6,6,0.1)
print('x,y range:',x.shape,y.shape)
'''
每一行显示[-6... 6]即为x一维矩阵(共120个元素)
简单理解，就是把x一维矩阵扩展（向下）成二维矩阵，行数对应于原先一维元素个数。
注意: 你不要只理解x,y组成二维坐标，这里实际上x为一个平面，y为一个平面
'''
X, Y = np.meshgrid(x,y) 
print('X,Y maps:',X.shape,Y.shape)
Z = himmelblau([X,Y])

flg = plt.figure('himmelblau')
ax = flg.gca(projection='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()

'''
pre对x,y的梯度，这里的x代表x,y
优化器对x,y
'''
x = torch.tensor([-4.,0],requires_grad=True) #需要优化的参数
optimizer = torch.optim.Adam([x],lr=1e-3) #参数传入，设置学习率
for step in range(20000):
    pred = himmelblau(x)  #求出pred值
    optimizer.zero_grad() #即将梯度初始化为零
    pred.backward() #pred向前求梯度 得到 pre对x的梯度 x' ,同理 y'
    optimizer.step() #优化器更新参数 x = x - lr*x' y = y - lr*y' [只是表达下更新参数的意思,并不是adam真实的算法]
    if step % 2000 == 0:
        print('step {}: x= {},f(x) = {}'.format(step,x.tolist(),pred.item()))


'''
f(x,y) = 0.0的点
f(3.0,2.0) f(-2.805118,3.131312) f(-3.779310,-3.283186) f(3.584428,-1.848126)
'''