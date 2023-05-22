import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

torch.manual_seed(311)

class DenseNet(nn.Module):
    def __init__(self, in_feature=3, out_feature=1000):
        super(DenseNet, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.conv = nn.Conv2d(in_channels=self.in_feature, out_channels=64, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        out = self.conv(x)
        return out

if __name__ == '__main__':
    x = torch.randn(3, 299, 299)
    model = DenseNet()
    out = model(x)
    summary(model, (3, 299, 299))