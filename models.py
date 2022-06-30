import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math


class BlockUnet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BlockUnet,self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
    def forward(self,x):
        out=self.relu(self.conv1(x))
        out=self.bn1(out)
        out=self.relu(self.conv2(out))
        return self.bn2(out)
        return out

class EncoderUnet(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super(EncoderUnet,self).__init__()
        self.enc_blocks = nn.ModuleList([BlockUnet(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)   
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks[:-1]:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        x=self.enc_blocks[-1](x)
        ftrs.append(x)
        return ftrs


class DecoderUnet(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super(DecoderUnet, self).__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        for upconv in self.upconvs:
            nn.init.kaiming_normal_(upconv.weight, mode='fan_in', nonlinearity='relu')
        self.dec_blocks = nn.ModuleList([BlockUnet(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.lastconv1 = nn.Conv2d(64,64,kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.lastconv1.weight, mode='fan_in', nonlinearity='relu')
        self.lastbn1=nn.BatchNorm2d(64)
        self.lastconv2 = nn.Conv2d(64,8,kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.lastconv2.weight, mode='fan_in', nonlinearity='relu')
        self.lastbn2=nn.BatchNorm2d(8)
        self.lastconv3=nn.Conv2d(8,3,kernel_size=1)
        nn.init.kaiming_normal_(self.lastconv3.weight, mode='fan_in', nonlinearity='relu')
        self.relu  = nn.ReLU()
    def forward(self, x, encodeur_features):
        for i in range(len(self.chs)-1):
            x  = self.upconvs[i](x)
            enc_ftrs = encodeur_features[i]
            x  = torch.cat([x, enc_ftrs], dim=1)
            x  = self.dec_blocks[i](x)
        x  = self.relu(self.lastconv1(x))
        x  = self.lastbn1(x)
        x  = self.relu(self.lastconv2(x))
        x  = self.lastbn2(x)
        return self.lastconv3(x)



class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder = EncoderUnet()
        self.decoder = DecoderUnet()
    def forward(self,x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        return out
    
