import torch as t
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

class ResidualBlock1x1v1(nn.Module):  #h=h/4,w=w/4   2
    def __init__(self,inchannel,outchannel,stride=1):
        super(ResidualBlock1x1v1, self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(in_channels=inchannel,out_channels=outchannel,kernel_size=1,stride=stride,padding=1,bias=False),#17*17
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3,stride=stride,padding=0) #8*8
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride,padding=1, bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3,stride=stride,padding=0)
            )
    def forward(self,x):
        out = self.left(x)

        out=out+self.shortcut(x)
        out = F.relu(out)
        return out
class ResidualBlock1x1v2(nn.Module): #h=h/2    4
    def __init__(self,inchannel,outchannel,stride=1):
        super(ResidualBlock1x1v2, self).__init__()
        self.left=nn.Sequential(
            nn.Conv2d(inchannel,outchannel,kernel_size=1,stride=stride,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel,outchannel,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut=nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),#128*16*16
                nn.BatchNorm2d(outchannel)
            )
    def forward(self,x):
        out=self.left(x)
        out+=self.shortcut(x)
        out=F.relu(out)
        return out

class ResidualBlock3x3(nn.Module):#h=h/2  4
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock3x3,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),#128*16*16
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),#128*16*16
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),#128*16*16
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class ResidualBlock5x5(nn.Module): #   4
    def __init__(self,inchannel, outchannel, stride=1):
        super(ResidualBlock5x5, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=5, stride=stride, padding=2, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self,x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
class InceptionRes(nn.Module):
    def __init__(self,ResidualBlock1x1v1,ResidualBlock1x1v2,ResidualBlock3x3,ResidualBlock5x5,inchannel,outchannel):
        super(InceptionRes, self).__init__()
        self.layer1x1v1=self.maker_layer(ResidualBlock1x1v1,inchannel,outchannel,2)#branch1
        self.layer1x1v2_1=self.maker_layer(ResidualBlock1x1v2,inchannel,inchannel,2)#branch2_1x1
        self.layer1x1v2_2=self.maker_layer(ResidualBlock1x1v2,inchannel,inchannel,1)#branch3_1x1
        self.layer1x1v2_3= self.maker_layer(ResidualBlock1x1v2, inchannel, outchannel,2)#branch4_1x1
        self.layer3x3_1=self.maker_layer(ResidualBlock3x3,inchannel,np.int((outchannel+inchannel)/2),2)#branch3_3x3_1
        self.layer3x3_2 = self.maker_layer(ResidualBlock3x3,np.int((outchannel+inchannel)/2), outchannel,2)#branch3_3x3_2
        self.layer5x5=self.maker_layer(ResidualBlock5x5,inchannel,outchannel,2)#branch2_5x5
        self.avg=nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
    def maker_layer(self,Block,inchannel,outchannel,stride):
        layers=[]
        layers.append(Block(inchannel,outchannel,stride))
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
        return t.cat(outputs,1)
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

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),#64*32*32
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.layer1 = self.make_layer(inceptionR,64,128)#128*8*8
        self.Conv2x2=nn.Sequential(nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())

        self.Conv2x2_2=nn.Sequential(nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())
        self.layer2 = self.make_layer(inceptionR,128,512)#2048*2*2
        self.fc1 = nn.Linear(2048, 1024)
        self.bn1=nn.BatchNorm1d(1024)
        self.fc2 =nn.Linear(1024,num_classes)

    def make_layer(self,Block,inchannel,outchannel):
        layers=[]
        layers.append(Block(inchannel,outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):#14
        out = self.conv1(x)#64*32*32         2
        out = self.layer1(out)#512*8*8       8
        out=self.Conv2x2(out)               #2
        out=self.Conv2x2_2(out)#128*8*8      2
        out=self.layer2(out)#2048*2*2        8
        out = F.avg_pool2d(out,2)#2048*1*1
        out = out.view(out.size(0), -1)
        out = self.fc1(out)#1024
        out=self.bn1(out)
        out=F.relu(out)
        out=F.dropout(out)
        out=self.fc2(out)#10
        #out=F.softmax(out,1)
        return out


def MyNet():
   return ResNet()
