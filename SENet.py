import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(311)

class SEBlock(nn.Module):
    def __init__(self, input_channel=256, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.input_channel = input_channel
        self.reduction_ratio = reduction_ratio
        self.output_channel = self.input_channel//self.reduction_ratio
        self.seblock = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Linear(in_features=self.input_channel, out_features=self.output_channel), 
            nn.ReLU(inplace=True), 
            nn.Linear(in_features=self.output_channel, out_features=self.input_channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        output1 = x
        print(output1.shape)
        output2 = self.seblock(x)
        print(output2.shape)
        output = torch.matmul(output1, output2)
        return output
    

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = SEBlock()
    print(model)
    output = model(x)