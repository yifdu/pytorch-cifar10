import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from FeatureExtractor import Extractor
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

exact_list=["fc","Conv5_4"]
def train(model,optimizer,criterion,trainloader,epoch,writer):
    model.train()
    running_loss = 0.0
    correct = 0.0
    total = 0.0
    length=0.0
    for step, (inputs, labels) in enumerate(trainloader):
        length=len(trainloader)
        inputs = Variable(inputs.to(device))
        labels = Variable(labels.to(device))  ####必须先cuda后Variable
        optimizer.zero_grad()
        total += labels.size(0)
        outs = model(inputs)
        loss = criterion(outs, labels)
        _,pred=torch.max(outs.data,1)
        #pred = outs.data.max(1, keepdim=True)[1]
        correct += pred.eq(labels.data.view_as(pred)).cpu().sum()

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print('[epoch:%d,iter:%d] Loss:%.03f|Acc:%.3f%%'
              % (epoch + 1, (step+1+epoch*length), running_loss / (step + 1), 100. *correct.numpy() / total))

    acc = correct.numpy() / total
    print('[%d train-epoch] Loss:%.3f acc:%.3f' % (epoch + 1,running_loss/length, acc))
    writer.add_scalar('train/acc',acc,epoch)
