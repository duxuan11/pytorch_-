#索引和切片
import torch
a = torch.rand(4,3,28,28) #4张图3个通道，大小28*28
print(a[0].shape,a[0,0].shape,a[0,0,2,4])
print("------------------------------------------")
#0
#1前俩张图片尺寸
print(a[:2].shape)
#2前俩张图片第一个通道的大小
print(a[:2,:1,:,:].shape)
print(a[:2,:1].shape)
#-1表示从最后一个元素取 它只有1个channel 可以-2:就是2个channel
print(a[:2,-1:,:,:].shape)
#隔行隔列采样 大小就是 4*3*14*14
print(a[:,:,0:28:2,0:28:2].shape)
print(a[:,:,::2,::2].shape)
print("------------------------------------------")
#如果我不是只想选某一个怎么办 (出现不连续 用不了:)
#表示对第二维，取第一个和第三个通道的数据
print(a.index_select(1,torch.tensor([0,2])).shape)
#第三维只取前8行
print(a.index_select(2,torch.arange(8)).shape)
print("------------------------------------------")
#...表示任意多的维度   ...可以替代任意多的:,
print(a[...].shape)
print(a[0,...].shape)
print(a[...,:8,:].shape)
print("------------------------------------------")
#使用掩码索引
x = torch.randn(3,4)
#大于0.5的置为1小于0.5置为0,true or false
mask = x.ge(0.5)
print(mask)
print(torch.masked_select(x,mask)) #选取所有大于0.5
print("------------------------------------------")
src = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
#take可以把多维数组改成1维，并取出1维数组中的元素
print(torch.take(src,torch.tensor([0,2,1])))