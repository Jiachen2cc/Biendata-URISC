import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
#针对二分类问题所写
class Focal_loss(nn.Module):
    def __init__(self,alpha = [0.25,0.75],gamma = 2,num_classes = 2,average = True):
        super(Focal_loss,self).__init__()
        self.average = average
        self.alpha = torch.FloatTensor(alpha)
        self.gamma = gamma
    
    def forward(self,x,labels):
        # x 预测的标签概率分布 [Channels,classes,row,col]
        # label 实际的标签    [channels,1,row,col]
        x = x*labels+(1-x)*(1-labels)
        
        print('x_min:{:.4f}'.format(torch.min(x)))
        alpha = self.alpha[labels].to(x.device)
        loss = -alpha*((1-x)**(self.gamma)*torch.log(x))
        if self.average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def Dice_loss(x,labels):
    intersect = (x*labels).sum()
    sum = x.sum() + labels.sum()
    return 1 - (2*intersect+1)/(sum + 1)
        


