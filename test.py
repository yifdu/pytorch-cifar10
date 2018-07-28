import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
device=torch.device("cuda" if torch.cuda.is_available()else "cpu")

def test(model,criterion,testloader,epoch,writer):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for step, (data, labels) in enumerate(testloader):
            inputs = Variable(data.to(device))
            labels = Variable(labels.to(device))
            outs = model(inputs)
            loss = criterion(outs, labels)
            total += labels.size(0)
            _,pred=torch.max(outs.data,1)
            #pred = outs.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum()
            running_loss += loss.item()

        acc = 100.*correct.numpy() / total
        avger_running_loss = running_loss / total
        print('[test-%d] loss: %.3f acc:%.3f%%' % (epoch + 1, avger_running_loss, acc))
        writer.add_scalar('test/acc',acc,epoch)