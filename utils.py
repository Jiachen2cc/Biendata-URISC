import cv2 as cv 
import numpy as np
import os 
import torch
import random
import time
import torch.nn.functional as F
from loader import lazy_loader,get_img_multi,output_img,get_img

#二分类标签转化为one_hot
def encode_onehot(labels):
    labels = F.one_hot(labels,num_classes = 2).permute(2,0,1).float()
    return labels
    
def f1_score(pred,label):

    TP = (pred*label).sum()
    FP = (pred*(1-label)).sum()
    FN = ((1-pred)*label).sum()
    
    print("TP:{},FP:{},FN:{}".format(TP,FP,FN))
    if (TP == 0 and FN == 0) or (TP == 0 and FP == 0):
        return 0
    P = TP/(TP + FP)
    R = TP/(TP + FN)
    return 2*P*R/(P+R)

def batch_f1_score(output,batch_label):

    batch_size = output.shape[0]
    score = 0
    for i in range(batch_size):
        pred = torch.ge(output[i,:,:,:],0.5)
        pred = pred.long()
        label = batch_label[i,:,:,:]
        s = f1_score(pred,label)
        if s == 0:
            return i
        score += s
    
    return score/batch_size
    

def divide_complex(dataset = 'two_stage_train',dst = 'two_stage_dataset',crop_num = 150,val_p = 0.1):
    
    path = os.path.dirname(os.path.dirname(__file__))
    out_path = os.path.join(path,dst)
    out_train_path = os.path.join(out_path,'train')
    out_train_label_path = os.path.join(out_path,'train_label')
    out_val_path = os.path.join(out_path,'val')
    out_val_label_path = os.path.join(out_path,'val_label')

    path = os.path.join(path,dataset)

    #load img
    train_path = os.path.join(path,'train')
    train_label_path = os.path.join(path,'train_label')
    train_dir = os.listdir(train_path)
    train_label_dir = os.listdir(train_label_path)
    for i in range(len(train_dir)):
        train_img = torch.FloatTensor(get_img_multi(os.path.join(train_path,train_dir[i])))
        train_label = torch.FloatTensor(get_img(os.path.join(train_label_path,train_label_dir[i])))
        print('begin random crop for {}th img'.format(i))
        for s in range(crop_num):
            x = np.random.randint(8935)
            y = np.random.randint(8934)
            img = train_img[x:x+1024,y:y+1024,:].numpy()
            label = train_label[x:x+1024,y:y+1024].numpy()
            
            num = i*crop_num + s
            if np.random.rand() > val_p:
                cv.imencode('.png',img)[1].tofile(os.path.join(out_train_path,'{}.png').format(num))
                cv.imencode('.png',label)[1].tofile(os.path.join(out_train_label_path,'{}.png').format(num))
            else:
                cv.imencode('.png',img)[1].tofile(os.path.join(out_val_path,'{}.png').format(num))
                cv.imencode('.png',label)[1].tofile(os.path.join(out_val_label_path,'{}.png').format(num))

def random_erasing(img,label = None,num_circle = 3 , rl = 25,rh = 75,
                       num_rec = 8 , xl = 25,xh = 75, yl = 25,  yh = 75,two_stage = False):
    
    mu = 0
    sigma = 10

    start = time.time()
    row,col,channel = img.shape[0],img.shape[1],img.shape[2]
    mean = np.mean(img)
    #generate circle mask
    for i in range(num_circle):
        r = random.randint(rl,rh)

        x = random.randint(r,row - r)
        y = random.randint(r,col - r)

        for j in range(x - r, x + r):
            for k in range( y - r ,y + r):
                if (j - x)**2 + (k - y)**2 <= r**2:
                    img[j][k] = mean + np.random.normal(mu,sigma)
                    if two_stage:
                        label[j][k] = 255
    
    #generate rectangle mask
    for i in range(num_rec):
        l = random.randint(xl,xh)
        w = random.randint(yl,yh)

        x = random.randint(0,row - l)
        y = random.randint(0,col - w)

        img[x:x+l,y:y+w,:] = mean + np.random.normal(mu,sigma,(l,w,channel))
        if two_stage:
            label[x:x+l,y:y+w] = 255
    '''
    cv.imencode('.png',img)[1].tofile(output_name+'img.png')
    cv.imencode('.png',label)[1].tofile(output_name+'label.png')
    '''
    end = time.time()

    print("time cost:{:.4f}".format(end - start))
    return img,label

def erasing(dataset,sum = 2000,two_stage = False):
    train_img,train_label,_,_,_ = lazy_loader(dataset,test = False)
    target = np.random.choice(range(len(train_img)),sum, replace = False)
    
    path = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(path,dataset)
  
    for num in target:
        img,label = random_erasing(get_img_multi(train_img[num]),get_img(train_label[num]),two_stage=two_stage)
        '''
        label = get_img_multi(train_label[num])
        print(label.shape)
        '''
        output_img(os.path.join(path,'train','{}_erased.png'.format(num)),img)
        if two_stage:
            output_img(os.path.join(path,'train_label','{}_erased.png'.format(num)),label)
        else:
            output_img(os.path.join(path,'train_label','{}_erased.png'.format(num)),get_img(train_label[num]))

    return True
def gamma_transform(dataset, p = 0.2):
    train_img,train_label,_,_,_ = lazy_loader(dataset,test = False)

    for num in range(len(train_img)):
        if np.random.rand() < p:
            img = get_img_multi(train_img[num]).astype(np.float64)
            img = img/255
            img = np.power(img,0.7)*255
            img = img.astype(np.uint8)
            output_img(train_img[num],img)


#roberts算子
def Robert(img):
    kernelx = np.array([[-1, 0], [0, 1]], dtype= int)
    kernely = np.array([[0, -1], [1, 0]], dtype= int)
    x = cv.filter2D(img, cv.CV_16S, kernelx)
    y = cv.filter2D(img, cv.CV_16S, kernely)
 
    #转uint8
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
 
    #加权和
    Roberts = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    print(1)
    print(Roberts)
    return Roberts




