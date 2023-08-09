import torch.nn as nn
import torch
from torch.nn.modules.batchnorm import BatchNorm2d

class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer,self).__init__()
        self.Conv_BN_RELU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels = out_ch,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_ch,out_channels = out_ch,kernel_size=3,padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsamlpe = nn.Sequential(
            nn.Conv2d(in_channels = out_ch,out_channels = out_ch,kernel_size=3,stride=2,padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU() 
        )
    def forward(self,x):
        out = self.Conv_BN_RELU_2(x)
        dout = self.downsamlpe(out)

        return out , dout

class UpsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UpsampleLayer,self).__init__()
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels = out_ch*2,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_ch*2,out_channels = out_ch*2,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(out_ch*2),
            nn.ReLU()
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch*2,out_channels = out_ch,kernel_size = 3, stride = 2,padding = 1,output_padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    
    def forward(self,x,out):
        # out:与上采样层进行cat
        x_out = self.Conv_BN_ReLU_2(x)
        x_out = self.upsample(x_out)
        cat_out = torch.cat((x_out,out),dim=1)

        return cat_out

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, stride = 1, padding=0, bias = False):
        super(Block, self).__init__()

        self.basic = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, padding, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_ch, out_ch, 3, 1, padding, bias = bias),
            nn.BatchNorm2d(out_ch),
            
        )
        self.downsample = None
        if stride > 1:
            self.downsample=nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = x
        out = self.basic(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = nn.ReLU(inplace = True)(out)
        return out

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=0, bias=False):
        super(Encoder, self).__init()

        self.block = nn.Sequential(
            Block(in_ch, out_ch, stride, padding, bias),
            Block(out_ch, out_ch, 1, padding, bias)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, output_padding=0,bias=False):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(in_ch//4),
            nn.LeakyReLU(0.1,inplace=True),
        )

        self.tp_conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch//4, in_ch//4, kernel_size, stride, padding, output_padding, bias=bias),
            nn.BatchNorm2d(in_ch//4),
            nn.LeakyReLU(0.1,inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch//4, out_ch, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1,inplace = True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)
        return x


