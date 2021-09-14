import torch
#数学操作
a = torch.rand(3,4)
b = torch.rand(4)

print(torch.all(torch.eq(a+b,torch.add(a,b))))
print(torch.all(torch.eq(a-b,torch.sub(a,b))))
print(torch.all(torch.eq(a/b,torch.div(a,b))))
print(torch.all(torch.eq(a*b,torch.mul(a,b))))
print("------------------------------------------")
''' 
* element-wise 对应元素相乘
matmul 是矩阵相乘
mm 只适用于2d
@ 简洁方法
'''
a = torch.tensor([[3.,3.],[3.,3.]])
b = torch.ones(2,2)
#mm 元素类型必须一致
print(torch.mm(a,b)) 
#matmul
print(torch.matmul(a,b))
#@
print(a@b)
print("------------------------------------------")
# 4*784 784*512 -> 4*512
#pytorch 常见写法 channel-out, channel-in
#因此x w的转置 
#.t只适用于2维
x = torch.rand(4,784)
w = torch.rand(512,784)
print((x@w.t()).shape)
print("------------------------------------------")
a = torch.rand(4,3,28,64)
b = torch.rand(4,1,64,28)
print(torch.matmul(a,b).shape)
print("------------------------------------------")
a = torch.full([2,2],3)
print(a.pow(2))
print(a**2)
aa = a**2
print(aa.sqrt()) #square root
print(aa.rsqrt()) #平方根的导数
print(aa**(0.5))
print(torch.exp(torch.ones(2,2)))
print(torch.log(a))
print("------------------------------------------")
#数字近似
a = torch.tensor(3.14)
#tensor(3.1400) tensor(4.) tensor(3.) tensor(0.1400) tensor(3.)
print(a.float(),a.ceil(),a.trunc(),a.frac(),a.round())

#w.grad.norm(2) #梯度二范数
print("------------------------------------------")
grad = torch.rand(2,3)*15
print(grad)
print(grad.max())
print(grad.median())
print(grad.clamp(10)) #小于10的都变为10
print(grad.clamp(0,9))#小于0的变为0，大于9的变为9