"""All trainable modules are CNNs or Dense layers; no attention or CTC."""
import torch
from torch import nn

def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU())

class Backbone(nn.Sequential):
    def __init__(self):super().__init__(block(1,32),nn.MaxPool2d(2),block(32,64),nn.MaxPool2d(2),block(64,128))

class SymbolCNN(nn.Module):
    def __init__(self,classes):
        super().__init__();self.backbone=Backbone();self.embed=nn.Sequential(nn.AdaptiveAvgPool2d((3,3)),nn.Flatten(),nn.Linear(128*9,128),nn.ReLU());self.output=nn.Linear(128,classes)
    def features(self,x):return self.embed(self.backbone(x))
    def forward(self,x):return self.output(self.features(x))

class RelationMLP(nn.Sequential):
    def __init__(self):super().__init__(nn.Linear(128*2+16,128),nn.ReLU(),nn.Dropout(.1),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,7))

class SymbolDetector(nn.Module):
    stride=4
    def __init__(self):
        super().__init__();self.backbone=Backbone()
        self.context=nn.Sequential(nn.AvgPool2d(4,ceil_mode=True),nn.Conv2d(128,64,3,padding=1),nn.ReLU(),
            nn.Conv2d(64,64,3,padding=4,dilation=4),nn.ReLU(),nn.Conv2d(64,64,3,padding=8,dilation=8),nn.ReLU())
        self.head=nn.Sequential(nn.Conv2d(192,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,5,1))
        nn.init.constant_(self.head[-1].bias[0],-2.19)
    def forward(self,x):
        f=self.backbone(x);context=torch.nn.functional.interpolate(self.context(f),size=f.shape[-2:],mode='bilinear',align_corners=False)
        z=self.head(torch.cat([f,context],1));return z[:,:1],torch.nn.functional.softplus(z[:,1:3]),z[:,3:5]
