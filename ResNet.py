import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(311)

class ResNet(nn.Module):
    def __init__(self, ):

        return 

if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)
    model = ResNet()
    output = model(x)
    print(output.shape)