import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(311)
# Paper: https://arxiv.org/abs/1409.1556
"""
Status: (230422) 논문읽고 코드 구현, ReLU 아직 구현 안함
"""

class VGG16_paper(nn.Module):
    
    def __init__(self):
        super().__init__()
        # self.conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(1, 1)) # Conv1d-1차원; Conv2d-2차원 (in_ch, out_ch, kernel)
        self.conv_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_5 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        
        # self.conv = nn.Conv2d(3, 3, 1) # a linear transformation of the input channels
        self.maxpool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(7*7*512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool(x)
        x = self.conv_2(x)
        x = self.maxpool(x)
        x = self.conv_3(x)
        x = self.conv_3_1(x)
        x = self.maxpool(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.maxpool(x)
        x = self.conv_5(x)
        x = self.conv_5(x)
        x = torch.flatten(self.maxpool(x)) # Flatten -> torch.flatten()
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=0) # activation function: softmax
        return x
    
if __name__ == '__main__':
    x = torch.rand(3, 224, 224) # input size (N, C, H, W)
    model = VGG16_paper()
    print(model)
    output = model(x)
    print('-'*50)
    print(output.shape)