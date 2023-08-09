from loader import load_data, output_img,get_img_multi,clean_dataset,get_img
import torch
import numpy as np
import cv2 as cv
import loader
from model import Linknet
import os
from utils import erasing,divide_complex,gamma_transform
import argparse
# src 中数据需要分布在 train,train_label,val,val_label,test 5个子文件夹中
# dst 需要提前创建 train,train_label,val,val_label,test 5个子文件夹
parser = argparse.ArgumentParser()
parser.add_argument('--src',default = 'stage2',
                    help = 'the dataset for the data need to be preprocessed')
parser.add_argument('--dst',default = 'stage2small',
                    help = 'the dataset where you want to store preprocessed data')
args = parser.parse_args()

def sliding_window(dataset='two_stage_train', dst = 'two_stage_dataset',val_p=0.1):

    path = os.path.dirname(os.path.dirname(__file__))

    out_path = os.path.join(path, dst)
    out_train_path = os.path.join(out_path, 'train')
    out_train_label_path = os.path.join(out_path, 'train_label')
    out_val_path = os.path.join(out_path, 'val')
    out_val_label_path = os.path.join(out_path, 'val_label')

    path = os.path.join(path, dataset)

    # load img
    train_path = os.path.join(path, 'train')
    train_label_path = os.path.join(path, 'train_label')
    train_dir = os.listdir(train_path)
    train_label_dir = os.listdir(train_label_path)
    
    cnt = 5000
    for i in range(len(train_dir)):
        print(os.path.join(train_path,train_dir[i]))
        train_img = torch.FloatTensor(get_img_multi(os.path.join(train_path,train_dir[i])))
        print(train_img.shape)
        train_label = torch.FloatTensor(get_img(os.path.join(train_label_path,train_label_dir[i])))
        width = 1024
        length = 1024
        r_bound = len(train_img) - width
        u_bound = len(train_img[0]) - length
        stride = 1024
        stepx = (r_bound + width) // stride + 1
        stepy = (u_bound + length) // stride + 1
        xchg = (width * stepx - len(train_img)) // (stepx-1) + 1
        ychg = (length * stepy - len(train_img[0])) // (stepy-1) + 1

        x = 0
        y = 0

        while x < r_bound:
            x_bound = x + width
            y = 0
            while y < u_bound:
                print("slicing!", x, y)
                y_bound = y + length

                img = train_img[x:x_bound, y:y_bound,:].numpy()
                label = train_label[x:x_bound,y:y_bound].numpy()

                if np.random.rand() > val_p:
                   cv.imencode('.png',img)[1].tofile(os.path.join(out_train_path,'{}.png').format(cnt))
                   cv.imencode('.png',label)[1].tofile(os.path.join(out_train_label_path,'{}.png').format(cnt))
                else:
                   cv.imencode('.png',img)[1].tofile(os.path.join(out_val_path,'{}.png').format(cnt))
                   cv.imencode('.png',label)[1].tofile(os.path.join(out_val_label_path,'{}.png').format(cnt))
                
                cnt += 1
                y = y + stride - ychg

            x = x + stride - xchg

    


if __name__ == '__main__':
    #sliding_window(args.src,args.dst)
    divide_complex(args.src,args.dst)
    #gamma_transform('two_dataset_1')

    erasing(args.dst,two_stage=True)
    clean_dataset(args.dst)
