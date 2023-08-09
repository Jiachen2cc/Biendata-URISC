import cv2 as cv
import os
import numpy as np
import torch

project_path = os.path.dirname(os.path.dirname(__file__))
def get_img(file_path,load_mode = 0):
    return cv.imdecode(np.fromfile(file_path,dtype = np.uint8),load_mode)

def get_img_multi(file_path):
    return cv.imdecode(np.fromfile(file_path,dtype = np.uint8),cv.IMREAD_COLOR)
    
def output_img(filepath,img,img_type = '.png'):
    cv.imencode(img_type,img)[1].tofile(filepath)

def preprocess_img(img):
    img = img.astype(np.float64)
    return img/127.5 -1

def load_data(dataset):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),dataset)

    #读取训练集图像数据
    train_img = []
    train_path = os.path.join(path,'train')
    for  name  in os.listdir(train_path):
        #由于路径中存在中文
        #print(torch.FloatTensor(get_img(os.path.join(train_path,name))).shape)
        train_img.append(torch.FloatTensor(get_img(os.path.join(train_path,name))))#.permute(2,0,1))
    
    #读取训练集图像分割groundtruth
    train_label = []
    train_label_path = os.path.join(path,'labels','train')
    
    for name in os.listdir(train_label_path):
        #print(torch.FloatTensor(get_img(os.path.join(train_label_path,name))).shape)
        train_label.append(torch.LongTensor(get_img(os.path.join(train_label_path,name))//255))#.permute(2,0,1))
    #读取评测集图像数据
    val_img = []
    val_path = os.path.join(path,'val',)

    for name in os.listdir(val_path):
        val_img.append(torch.FloatTensor(get_img(os.path.join(val_path,name))))#.permute(2,0,1))
    
    return train_img,train_label,val_img
'''
utils.divide_complex('complex')
'''

def lazy_loader(dataset,test = False):
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),dataset)

    #读取训练集图像数据
    train_img = []
    train_path = os.path.join(path,'train')
    for  name  in os.listdir(train_path):
        #由于路径中存在中文
        #print(torch.FloatTensor(get_img(os.path.join(train_path,name))).shape)
        train_img.append(os.path.join(train_path,name))
    
    #读取训练集图像分割groundtruth
    train_label = []
    train_label_path = os.path.join(path,'train_label')
    
    for name in os.listdir(train_label_path):
        #print(torch.FloatTensor(get_img(os.path.join(train_label_path,name))).shape)
        train_label.append(os.path.join(train_label_path,name))#.permute(2,0,1))

    #读取评测集图像数据
    val_img = []
    val_path = os.path.join(path,'val')
    for name in os.listdir(val_path):
        val_img.append(os.path.join(val_path,name))#.permute(2,0,1))

    #读取评测集图像分类groundtruth
    val_label = []
    val_label_path = os.path.join(path,'val_label')
    for name in os.listdir(val_label_path):
        val_label.append(os.path.join(val_label_path,name))
    
    test_img = []
    if test:
        test_path = os.path.join(path,'test')
    
        for name in os.listdir(test_path):
            test_img.append(os.path.join(test_path,name))

    return train_img,train_label,val_img,val_label,test_img

def clean_dataset(dataset):
    train_img,train_label,val_img,val_label,_  = lazy_loader(dataset)

    for num in range(len(train_img)):
        label = 1 - get_img(train_label[num])//255
        if np.sum(label) < 5000:
            os.remove(train_label[num])
            os.remove(train_img[num])

    for num in range(len(val_img)):
        label = 1 - get_img(val_label[num])//255
        if np.sum(label) < 5000:
            os.remove(val_label[num])
            os.remove(val_img[num])
'''
if __name__ == '__main__':
    clean_dataset('two_stage_dataset')
'''
