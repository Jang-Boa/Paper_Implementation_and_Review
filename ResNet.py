import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(311)

class BottleNeck(nn.Module):
    """ BottleNeck의 핵심은 1x1 Conv. """
    def __init__(self, in_features, out_features, downsample=None):
        super(BottleNeck, self).__init__()
        self.layer_ = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0), # 1x1 conv
            nn.BatchNorm2d(out_features),
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=1, padding=1), # 3x3 conv
            nn.BatchNorm2d(out_features),
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0), # 1x1 conv
            nn.BatchNorm2d(out_features),
        )

        self.downsample = downsample
        self.relu = nn.ReLU()
    
    def forward(self, x):
        identity = x.clone() # copy tensor 
        out = self.layer_(x)

        if self.downsample is not None: # downsample 
            identity = self.downsample(x)
        
        out += identity # shortcut connection simply perform identity mapping
        out = self.relu(x)
        return x



class ResNetPlain(nn.Module):# 34-layer plain
    def __init__(self, num_channels=3):
        super(ResNetPlain, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self.conv3x3(64, 64, )
        
    def conv3x3(self, in_features, out_features, kernel_size=3, stride=1):
        layer = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_features),
            nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(out_features),
        )
    
    def forward(self, x):
        out = self.layer0(x)
        out = self.layer1(x)
        return out

if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)
    model = ResNetPlain()
    output = model(x)
    print(output.shape)