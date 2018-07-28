import torch as t
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

class ResidualBlock1x1v1(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1):
        super(ResidualBlock1x1v1, self).__init__()
        self.left=nn.Sequential(
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=1,stride=stride,padding=0,bias=False),#17*17
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=outchannel,out_channels=inchannel,kernel_size=1,stride=stride,padding=0,bias=False)
        )

    def forward(self,x):
        out = self.left(x)
        out=out+x
        return out
class ResidualBlock1x1v2(nn.Module): #h=h/2
    def __init__(self,inchannel,outchannel,stride=1):
        super(ResidualBlock1x1v2, self).__init__()
        self.left=nn.Sequential(
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,padding=0,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,inchannel,kernel_size=1,stride=1,padding=0,bias=False)
        )

    def forward(self,x):
        out=self.left(x)
        out=out+x
        return out

class ResidualBlock3x3(nn.Module):#h=h/2
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock3x3,self).__init__()
        self.left = nn.Sequential(
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),#128*16*16
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, inchannel, kernel_size=3, stride=1, padding=1, bias=False),#128*16*16
        )


    def forward(self, x):
        out = self.left(x)
        out += x
        return out
class ResidualBlock5x5(nn.Module):
    def __init__(self,inchannel, outchannel, stride=1):
        super(ResidualBlock5x5, self).__init__()
        self.left = nn.Sequential(
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, outchannel, kernel_size=5, stride=stride, padding=2, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, inchannel, kernel_size=5, stride=1, padding=2, bias=False),
        )

    def forward(self,x):
        out = self.left(x)
        out +=x
        return out
class InceptionRes(nn.Module):
    def __init__(self,ResidualBlock1x1v1,ResidualBlock1x1v2,ResidualBlock3x3,ResidualBlock5x5,inchannel,outchannel):
        super(InceptionRes, self).__init__()
        self.layer1x1v1=self.maker_layer(ResidualBlock1x1v1,inchannel,outchannel,1)#branch1
        self.layer1x1v2_1=self.maker_layer(ResidualBlock1x1v2,inchannel,outchannel,1)#branch2_1x1
        self.layer1x1v2_2=self.maker_layer(ResidualBlock1x1v2,inchannel,outchannel,1)#branch3_1x1
        self.layer1x1v2_3= self.maker_layer(ResidualBlock1x1v2, inchannel,outchannel,1)#branch4_1x1
        self.layer3x3_1=self.maker_layer(ResidualBlock3x3,inchannel,outchannel,1)#branch3_3x3_1
        self.layer3x3_2 = self.maker_layer(ResidualBlock3x3,inchannel,outchannel,1)#branch3_3x3_2
        self.layer5x5=self.maker_layer(ResidualBlock5x5,inchannel,outchannel,1)#branch2_5x5
        self.avg=nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.SEblock=self.maker_layer_SEBlock(SEBlock,4,inchannel)
    def maker_layer(self,Block,inchannel,outchannel,stride):
        layers=[]
        layers.append(Block(inchannel,outchannel,stride))
        return nn.Sequential(*layers)
    def maker_layer_SEBlock(self,Block,way,channel):
        layers=[]
        layers.append(Block(way,channel))
        return nn.Sequential(*layers)
    def forward(self,x):
        out1=self.layer1x1v1(x)
        out2=self.layer1x1v2_1(x)
        out2=self.layer5x5(out2)
        out3=self.layer1x1v2_2(x)
        out3=self.layer3x3_1(out3)
        out3=self.layer3x3_2(out3)
        out4=self.avg(x)
        out4=self.layer1x1v2_3(out4)
        outputs = [out1,out2,out3,out4]
        outputs=self.SEblock(outputs)
        return outputs
class inceptionR(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(inceptionR, self).__init__()
        self.struct=self.make_layer(InceptionRes,ResidualBlock1x1v1,ResidualBlock1x1v2,ResidualBlock3x3,ResidualBlock5x5,inchannel,outchannel)
    def forward(self,x):
        out=self.struct(x)
        return out
    def make_layer(self,BB,Block1,Block2,Block3,Block4,inchannel,outchannel):
        layers=[]
        layers.append(BB(Block1,Block2,Block3,Block4,inchannel,outchannel))
        return nn.Sequential(*layers)
class SEBlock(nn.Module):
    def __init__(self,way,channel):
        super(SEBlock, self).__init__()
        self.channel=channel
        self.way=way
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc_in_SE=nn.Sequential(
            nn.Linear(in_features=channel,out_features=channel//4,bias=None),
            nn.ReLU(),
            nn.Linear(in_features=channel//4,out_features=1,bias=None),
            nn.Sigmoid()
        )
        self.fc=nn.Sequential(
            nn.Linear(self.way,4*self.way),
            nn.ReLU(),
            nn.Linear(4*self.way,self.way),
            nn.Sigmoid()

        )

    def forward(self,x):
        size=x[0].size()
        num=size[0]
        channel=size[1]
        W=size[2]
        H=size[3]
        X=t.zeros((num,self.way,channel,W,H))
        XX=t.zeros((num,self.way,channel,W,H)).cuda()
        xx=[]
        for i in range(self.way):
            XX[:,i,:,:,:]=x[i]
            x[i]=self.avg_pool(x[i])
            x[i]=x[i].view(x[i].size(0),-1)
            xx.append(self.fc_in_SE(x[i]))

        y=t.cat(xx,1)
        y=self.fc(y).view(num,self.way,1,1,1)
        Z=XX*y
        out=[]
        for i in range(self.way):
            out.append(Z[:,i,:,:,:])

        return t.cat(out,1)




class SENet(nn.Module):
    def __init__(self,inchannel,reduction=16):
        super(SENet, self).__init__()
        self.inchannel=inchannel
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Linear(inchannel,inchannel//reduction),
                              nn.ReLU(inplace=True),
                              nn.Linear(inchannel//reduction,inchannel),
                              nn.Sigmoid())
    def forward(self,x):
        b,c,_,_=x.size()
        y=self.avg_pool(x).view(b,c)
        y=self.fc(y).view(b,c,1,1)
        return x*y

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),#64*32*32
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.weight1=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1,bias=False)#128*16*16
        self.layer1 = self.make_layer(inceptionR,128,256)#512*16*16
        self.SENet1=self.make_SENet(SENet,512)
        self.Conv2x2=nn.Sequential(nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,padding=1,stride=2),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU())#1024*8*8

        self.Conv2x2_2=nn.Sequential(nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=3,padding=1,stride=2),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU())#512*4*4
        self.weight2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1,
                                 bias=False)  # 256*2*2

        self.layer2 = self.make_layer(inceptionR,256,512)#1024*2*2
        self.SENet2=self.make_SENet(SENet,1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1=nn.BatchNorm1d(512)
        self.fc2 =nn.Linear(512,num_classes)

    def make_layer(self,Block,inchannel,outchannel):
        layers=[]
        layers.append(Block(inchannel,outchannel))
        return nn.Sequential(*layers)
    def make_SENet(self,SENet,inchannel):
        layer=[]
        layer.append(SENet(inchannel,reduction=16))
        return nn.Sequential(*layer)
    def forward(self, x):#14
        out = self.conv1(x)#64*32*32         2
        out=self.weight1(out)#128*16*16
        out = self.layer1(out)#512*16*16       4
        out=self.SENet1(out)
        out=self.Conv2x2(out) #1024*8*8              #2
        out=self.Conv2x2_2(out)#512*4*4      2
        out=self.weight2(out)#256*2*2
        out=self.layer2(out)#1024*2*2        4
        out=self.SENet2(out)
        out = F.avg_pool2d(out,2)#1024*1*1
        out = out.view(out.size(0), -1)
        out = self.fc1(out)#1024
        out=self.bn1(out)
        out=F.relu(out)
        out=F.dropout(out)
        out=self.fc2(out)#10
        #out=F.softmax(out,1)
        return out


def MyNet3():
   return ResNet()
