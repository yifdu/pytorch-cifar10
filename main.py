import torch as t

from torchsummary import summary
import torchvision as tv
from torch.utils import data
from torchvision import transforms as transforms
import os
from train import train
from test import test
import torch.nn as nn
from vgg16 import Vgg
from MyNet import MyNet
from MyNet3 import MyNet3
from resnet import ResNet18
from resnet18 import ResNet18
import torch.optim as optim
from tensorboardX import SummaryWriter
Num_epoch=1500
Batch_size=128
vgg16_pkl='./checkpoints/resnet_net1_params.pkl'
resnet18_pkl='./checkpoints/resnet_net_params.pkl'
device=t.device("cuda" if t.cuda.is_available() else "cpu")
LR=0.01
def weights_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        #m.bias.data.fill_(0)
    elif(isinstance(m,nn.Linear)):
        nn.init.xavier_normal_(m.weight.data)
        #m.bias.data.fill_(0)

def data_processing():
    transform_train=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    transform_test=transforms.Compose([transforms.Resize(32),transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))])
    train=tv.datasets.CIFAR10('./data',train=True,transform=transform_train,download=True)
    test=tv.datasets.CIFAR10('./data',train=False,transform=transform_test,download=True)
    trainloader=data.DataLoader(train,batch_size=Batch_size,shuffle=True,num_workers=4)
    testloader=data.DataLoader(test,batch_size=100,shuffle=True,num_workers=4)
    return trainloader,testloader


if __name__=="__main__":
    writer=SummaryWriter('log')
    trainloader,testloader=data_processing()
    net=ResNet18()
    #net=ResNet18()
    net.apply(weights_init)
    net.to(device=device)
    summary(net,(3,32,32))#类keras 的参数统计函数
    if (os.path.exists(vgg16_pkl)):
        net.load_state_dict(t.load(vgg16_pkl))
    criterion = nn.CrossEntropyLoss()
    # optimizer=optim.Adam(net.parameters(),lr=0.001)
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                          weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
    lr_sch = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    print("Start Training")

    for epoch in range(Num_epoch):
        print('\nEpoch:%d'%(epoch+1))
        train(net,optimizer,criterion,trainloader,epoch,writer)
        print("Waiting Test!")
        test(net,criterion,testloader,epoch,writer)
        t.save(net.state_dict(),vgg16_pkl)
        lr_sch.step()

