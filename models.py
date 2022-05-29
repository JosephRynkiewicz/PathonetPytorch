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
    
#Pathonet

class ResidualDilatedXception(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, t="e"):
        super(ResidualDilatedXception, self).__init__()
        self.t=t
        self.leakyrelu=nn.LeakyReLU(negative_slope=0.3)
        if self.t=="d":
            self.convd1=nn.Conv2d(in_ch,out_ch,kernel_size=1,padding='same',bias=False)
            nn.init.orthogonal_(self.convd1.weight)
            self.bnd1=nn.BatchNorm2d(out_ch)
            self.convd2=nn.Conv2d(out_ch,out_ch,kernel_size=1,padding='same',bias=False)
            nn.init.orthogonal_(self.convd2.weight)
            self.bnd2=nn.BatchNorm2d(out_ch)
            in_ch=out_ch
        self.convA1=nn.Conv2d(in_ch,out_ch,kernel_size=3,padding='same',bias=False)
        nn.init.orthogonal_(self.convA1.weight)
        self.bnA1=nn.BatchNorm2d(out_ch)
        self.convA2=nn.Conv2d(out_ch,out_ch,kernel_size=3,padding='same',bias=False)
        nn.init.orthogonal_(self.convA2.weight)
        self.bnA2=nn.BatchNorm2d(out_ch)
        self.convA3=nn.Conv2d(in_ch,out_ch,kernel_size=3,dilation=4,padding='same',bias=False)
        nn.init.orthogonal_(self.convA3.weight)
        self.bnA3=nn.BatchNorm2d(out_ch)
        self.convA4=nn.Conv2d(out_ch,out_ch,kernel_size=3,dilation=4,padding='same',bias=False)
        nn.init.orthogonal_(self.convA4.weight)
        self.bnA4=nn.BatchNorm2d(out_ch)
        self.bnlast=nn.BatchNorm2d(out_ch)
    def forward(self,x):
        if self.t=="d":
            x=self.convd1(x)
            x=self.leakyrelu(self.bnd1(x))
            x=self.convd2(x)
            x=self.leakyrelu(self.bnd2(x))
        out1=self.convA1(x)
        out1=self.leakyrelu(self.bnA1(out1))
        out1=self.convA2(out1)
        out1=self.leakyrelu(self.bnA2(out1))
        out2=self.convA3(x)
        out2=self.leakyrelu(self.bnA3(out2))
        out2=self.convA4(out2)
        out2=self.leakyrelu(self.bnA4(out2))
        if self.t=="e":
            x=torch.cat([x,x], dim=1)
        out=torch.add(out1,x)
        out=torch.add(out,out2)
        out=self.leakyrelu(self.bnlast(out))
        return out


class PathoNetEncoder(nn.Module):
    def __init__(self):
        super(PathoNetEncoder, self).__init__()
        self.leakyrelu=nn.LeakyReLU(negative_slope=0.3)
        self.pool2d=nn.MaxPool2d(kernel_size=2)
        self.dp=nn.Dropout(p=0.1)
        self.conv1=nn.Conv2d(3,16,kernel_size=3,padding='same',bias=False)
        nn.init.orthogonal_(self.conv1.weight)
        self.bn1=nn.BatchNorm2d(16)
        self.conv2=nn.Conv2d(16,16,kernel_size=3,padding='same',bias=False)
        nn.init.orthogonal_(self.conv2.weight)
        self.bn2=nn.BatchNorm2d(16)
        self.rdx1=ResidualDilatedXception(16,32)
        self.rdx2=ResidualDilatedXception(32,64)
        self.rdx3=ResidualDilatedXception(64,128)
        self.rdx4=ResidualDilatedXception(128,256)
    def forward(self,x):
        out1=self.conv1(x)
        out1=self.leakyrelu(self.bn1(out1))
        out1=self.conv2(out1)
        out1=self.leakyrelu(self.bn2(out1))
        out2=self.pool2d(out1)
        out2=self.rdx1(out2)
        out3=self.pool2d(out2)
        out3=self.rdx2(out3)
        out4=self.pool2d(out3)
        out4=self.rdx3(out4)
        out5=self.pool2d(out4)
        out5=self.dp(out5)
        out5=self.rdx4(out5)
        out5=self.dp(out5)
        return out1, out2, out3, out4, out5

class PathoNetDecoder(nn.Module):
    def __init__(self):
        super(PathoNetDecoder, self).__init__()
        self.leakyrelu=nn.LeakyReLU(negative_slope=0.3)
        self.relu=nn.ReLU()
        self.upsample=nn.Upsample(scale_factor=2)
        self.rdx1=ResidualDilatedXception(256,128,t="d")
        self.rdx2=ResidualDilatedXception(256,64,t="d")
        self.rdx3=ResidualDilatedXception(128,32,t="d")
        self.rdx4=ResidualDilatedXception(64,16,t="d")
        self.conv1=nn.Conv2d(32,16,kernel_size=3,padding='same',bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.conv2=nn.Conv2d(16,16,kernel_size=3,padding='same',bias=False)
        self.bn2=nn.BatchNorm2d(16)
        self.conv3=nn.Conv2d(16,8,kernel_size=3,padding='same',bias=False)
        self.bn3=nn.BatchNorm2d(8)
        self.conv4=nn.Conv2d(8,3,kernel_size=1)
    def forward(self,out1,out2,out3,out4,out5):
        up1=self.upsample(out5)
        up1=self.rdx1(up1)
        m1=torch.cat([out4,up1],dim=1)
        up2=self.upsample(m1)
        up2=self.rdx2(up2)
        m2=torch.cat([out3,up2],dim=1)
        up3=self.upsample(m2)
        up3=self.rdx3(up3)
        m3=torch.cat([out2,up3],dim=1)
        up4=self.upsample(m3)
        up4=self.rdx4(up4)
        m4=torch.cat([out1,up4],dim=1)
        out=self.conv1(m4)
        out=self.leakyrelu(self.bn1(out))
        out=self.conv2(out)
        out=self.leakyrelu(self.bn2(out))
        out=self.conv3(out)
        out=self.leakyrelu(self.bn3(out))
        out=self.conv4(out)
        return self.relu(out)


class PathoNet(nn.Module):
    def __init__(self):
        super(PathoNet, self).__init__()
        self.encoder=PathoNetEncoder()
        self.decoder=PathoNetDecoder()
    def forward(self,x):
        out1, out2, out3, out4, out5 = self.encoder(x)
        out=self.decoder(out1, out2, out3, out4, out5)
        return out
        
        
