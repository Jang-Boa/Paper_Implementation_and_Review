import torch
import torch.nn as nn
from torchsummary import summary

torch.manual_seed(311)

class EfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNet, self).__init__()
        
    def forward(self, x):
        return out 

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = EfficientNet()
    output = model(x)