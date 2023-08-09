from loader import project_path,lazy_loader,output_img
import torch
import numpy
import cv2 as cv
import loader
from model import Linknet
import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset',default = 'complex',
                    help = 'the dataset for big graph you want to predict')
parser.add_argument('--two_stage',action = 'store_true',default = False,
                    help = 'one_stage or two_stage')
args = parser.parse_args()
net1 = Linknet()
net1.load_state_dict(torch.load('first_stage/linknet_leaky_first_stagept'))

net2 = Linknet()
net2.load_state_dict(torch.load('second_stage/linknet_leaky_second_stage1800.pt'))

def tta_predict(img,net): # test time argumnentation 
    img2 = torch.flip(img, dims=[2])
    img3 = torch.flip(img, dims=[3])

    out1 = net(img)
    out2 = torch.flip(net(img2), dims = [2])
    out3 = torch.flip(net(img3), dims = [3])

    out4 = net(img.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    out5 = torch.flip(net(img2.permute(0, 1, 3, 2)).permute(0, 1, 3, 2), dims = [2])
    out6 = torch.flip(net(img3.permute(0, 1, 3, 2)).permute(0, 1, 3, 2), dims = [3])

    return (out1 + out2 + out3 + out4 + out5 + out6) / 6

def two_stage_generate(net,img):#生成二阶段的训练图

    img2 = torch.flip(img,dims = [2])
    img3 = torch.flip(img,dims = [3])

    out1 = net(img)
    out2 = torch.flip(net(img2), dims = [2])
    out3 = torch.flip(net(img3), dims = [3])
    
    out = torch.squeeze(torch.stack([out1,out2,out3],dim = 4))
    print(out.shape)

    return out



def sliding_window(path,two_stage):#膨胀预测
    train_img = torch.FloatTensor(loader.get_img_multi(path))
    
    width,length = 2048,2048
    r_bound = len(train_img[0])
    u_bound = len(train_img)
    print(r_bound, u_bound)
    stride = 1024
    #stepx = (r_bound + length) // stride + 1
    #stepy = (u_bound + width) // stride + 1
    x = 0
    y = 0
    output_img = torch.ones(len(train_img), len(train_img[0])).long()

    extendx_flag = 0
    while x < u_bound:
        x_bound = x + width
        extendx_flag = 0
        if x_bound > u_bound:
            x_bound = u_bound
            extendx_flag = 1
        y = 0
        while y  < r_bound:
            test_img = torch.Tensor(width, length,3)
            extendy_flag = 0
            y_bound = y + length
            if(y_bound > r_bound):
                y_bound = r_bound
                extendy_flag = 1

            disx = 0
            disy = 0
            
            if extendx_flag == 1 and extendy_flag == 0:
                #test_img[x_bound-x:width, 0:y_bound-y] = torch.ones(x+width-x_bound, y_bound - y)
                disx = x + width - x_bound
                test_img[0:width, 0:length] = train_img[x-disx:x_bound, y:y_bound]
            elif extendy_flag == 1 and extendx_flag == 0:
                disy = y + length - y_bound
                test_img[0:width, 0:length] = train_img[x:x_bound, y-disy:y_bound]
            elif extendx_flag == 1 and extendy_flag == 1:
                disx = x + width - x_bound
                disy = y + length - y_bound
                test_img[0:width, 0:length] = train_img[x-disx:x_bound, y-disy:y_bound]
            else:
                test_img[0:x_bound-x, 0:y_bound-y] = train_img[x:x_bound, y:y_bound]
            

            
            test_img = torch.unsqueeze(test_img.permute(2,0,1),dim = 0).float()
            '''
            #generate image for stage 2
            output = two_stage_generate(net1,test_img/127.5 - 1)
            pred = (output*255).long()
            '''
            if two_stage:
                output = tta_predict(net2,test_img/127.5 -1)
            else:
                output = tta_predict(net1,test_img/127.5 -1)
            pred = torch.ge(output,0.5).long()
            pred = torch.squeeze(pred)
            
            '''
            pred = 255*(1-torch.squeeze(pred))
            # test two stage
            pred = torch.unsqueeze(pred.permute(2,0,1),dim = 0)
            pred = pred.float() #这里是训练问题，忘记加preprocess_img了
            output = net2(pred)
            pred = torch.ge(output,0.5).long()
            pred = torch.squeeze(pred)
            '''

            output_img[x+100:x_bound-100, y+100:y_bound-100] = pred[disx+100:width-100, disy+100:length-100]    ##doubtful!


            if x == 0:
                output_img[0:100, y+100:y_bound-100] = pred[0:100, disy+100:length - 100]
            if y == 0:
                output_img[x+100:x_bound - 100, 0:100] = pred[disx+100:width - 100, 0:100]
            if x == 0 and y == 0:
                output_img[0:100, 0:100] = pred[0:100, 0:100]
            if x_bound == u_bound:
                output_img[x_bound-100:x_bound, y+100:y_bound - 100] = pred[width-100:width, disy+100:length-100]
            if y_bound == r_bound:
                output_img[x+100:x_bound-100, y_bound-100:y_bound] = pred[disx+100:width-100, length-100:length]
            if x_bound == u_bound and y_bound == r_bound:
                output_img[x_bound-100:x_bound,y_bound-100:y_bound] = pred[width-100:width,length-100:length]
            if x_bound == u_bound and y == 0:
                output_img[x_bound-100:x_bound, 0:100] = pred[width-100:width, 0:100]
            if x == 0 and y_bound == r_bound:
                output_img[0:100, y_bound-100:y_bound] = pred[0:100, length-100:length]
            
            y = y + stride

        x = x + stride

    return output_img

if __name__ == "__main__":
    
    train_img,_,val_img,_,test_img = lazy_loader(args.dataset,test = True)
    
    for name in test_img:
        
        result = 255 - sliding_window(name,two_stage = args.two_stage)
        result = result.numpy()

        x = name.split('/')[-1]
        path = os.path.join(project_path, 'pred','stage1_test',x)
        print(path)
        output_img(path, result)
    





    