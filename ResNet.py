import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(311)

class ResNet(nn.Module):# 34-layer plain
    def __init__(self, ):
        encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(634, )
        )
        return 

if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)
    model = ResNet()
    output = model(x)
    print(output.shape)