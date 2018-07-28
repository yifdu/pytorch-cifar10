import torch.nn as nn
import matplotlib.pyplot as plt
class FeatureExtractor(nn.Module):
    def __init__(self,submodule,extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule=submodule
        self.extracted_layers=extracted_layers
    def forward(self,x):
        outputs=[]
        for name,module in self.submodule._modules.items():
            if name is "fc": x=x.view(x.size(0),-1)
            x=module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs

def Extractor(modle,exact_list,img):
    myexactor=FeatureExtractor(modle,exact_list)
    x=myexactor(img)
    for i in range(64):
        ax=plt.subplot(8,8,i+1)
        ax.set_title('sample#{}'.format(i))
        ax.axis('off')
        plt.imshow(x[0].cpu().data.numpy()[0,i,:,:],cmap='jet')
    plt.show()