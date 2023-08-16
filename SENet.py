import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(311)

class SEBlock(nn.Module):
    def __init__(self, input_channel=256, r=16):
        super(SEBlock, self).__init__()
        self.input_channel = input_channel
        self.reduction_ratio = r
        self.output_channel = self.input_channel//self.reduction_ratio
        self.squeeze = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.excitation = nn.Sequential(
            nn.Linear(in_features=self.input_channel, out_features=self.output_channel), 
            nn.ReLU(inplace=True), 
            nn.Linear(in_features=self.output_channel, out_features=self.input_channel),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        output1 = x
        print(output1.shape)
        output2 = self.squeeze(x)
        print(output2.shape)
        output2 = torch.flatten(output2, 1)
        print(output2.shape)
        output2 = self.excitation(output2)
        print(output2.shape)
        output2 = output2.view(output2.shape[0], output2.shape[1], 1, 1)
        print(output2.shape)
        output = output1*output2 # channel-wise multiplication
        print(output.shape)
        return output
    

if __name__ == "__main__":
    x = torch.randn(2, 256, 224, 224)
    model = SEBlock()
    print(model)
    output = model(x)
    # print(output)