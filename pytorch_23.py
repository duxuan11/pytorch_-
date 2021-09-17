import torch
import torch.nn
import torch.nn.functional as F
'''
测试方法 1 准确度的小测试
'''

#假设有4张图片 每张有10个预测值
logits = torch.rand(4,10) 

pred = F.softmax(logits,dim=1)
print("pred       : {}".format(pred))
pred_label = pred.argmax(dim = 1)
print("pred_label : {}".format(pred_label))
print("argmax     : {}".format(logits.argmax(dim=1)))

label = torch.tensor([9,3,2,4])
correct = torch.eq(pred_label,label)
print("correct    : {}".format(correct))
print("Accuracy   : {}".format(correct.sum().float().item()/4))

