#tensor维度变换
'''
1.View/reshape (view 0.3版本)  变换
2.Squeeze/unsqueeze 删减维度和增加维度
3.Transpose/t/permute 矩阵得转置 
4.Expand/repeat 维度扩展
'''
import torch
a = torch.rand(4,1,28,28)
#对于4张图片，每张图片像素合在一起 用784维表示
print(a.view(4,28*28))
#图片和通道和行混在一起
print(a.view(4*28,28).shape)
'''
b = a.view(4,784)
有可能会破坏图片原有的维度信息，如果不额外记忆就恢复不了
'''
print("------------------------------------------")
print(a.shape)
#给这个数组插入一个维度(组别)
#参数是0，1，2，3，4 4个位置插入维度
print(a.unsqueeze(0).shape)
'''
torch.Size([1, 4, 1, 28, 28])
'''
print(a.unsqueeze(-1).shape)
'''
torch.Size([4, 1, 28, 28, 1])
'''
a = torch.tensor([1.2,2.3])
print(a.unsqueeze(-1))

#案例
#bias
b = torch.rand(32)
#图片
f = torch.rand(4,32,1,1)

b = b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
'''
print(f)
print('\n')
print(b)
print('\n')
print(b+f)
'''
print(b.squeeze().shape) #他所有维度为1的维度挤压掉
print(b.squeeze(-1).shape) #挤压掉最后一个维度

print("------------------------------------------")
#Expand扩张，但不复制数据
#Repeat重复，且复制数据
a = torch.rand(4,32,1,1)#维度di必须与原来一致，且扩张 要么与原来相同，要是从1扩张
print(a.expand(4,32,14,14))
#print(a.expand(4,64,14,14))  报错 64 想扩张必须是1
'''
4,32,1,1 - > 1,1,14,14 repeat里面是指复制的次数
'''
print(a.repeat(1,1,14,14).shape)
print("------------------------------------------")
a = torch.rand(4,3,32,32)
print(a)
#转置
#.t只能使用于2d的矩阵
#contiguous()是在内存里开辟一块新的地方存放转置后的矩阵
#用reshape代替view解决内存不连续的问题
#这个是不对的
a1 = a.transpose(1,3).contiguous().reshape(4,3*32*32).reshape(4,3,32,32)
print(a1.shape)#.shape)
a2 = a.transpose(1,3).contiguous().reshape(4,3*32*32).reshape(4,32,32,3).transpose(1,3)
print(a2.shape)
#.shape
print(torch.all(torch.eq(a1,a2)))
#torch.eq(a2,a) 它会比较每一处数据返回一个true or false矩阵
#而加上all则判断全部是否相同
print(torch.all(torch.eq(a,a2)))
print("------------------------------------------")
#permute相当于多次transpose
b = torch.rand(4,3,28,32)
b.transpose(1,3).shape
# b c h w -> b w h c -> b h w c
print(b.transpose(1,3).transpose(1,2).shape)
#直接用permute  参数是/原来的列号
print(b.permute(0,2,3,1).shape)

