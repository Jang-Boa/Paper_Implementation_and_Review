import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(311)

class GoogLeNet(nn.Module):
    def __init__(self, ):
        super(GoogLeNet, self).__init__()

    def forward(self, x):
        return out

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = GoogLeNet()
    output = model(x)
    print(model)