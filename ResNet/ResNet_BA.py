import torch
from torch import nn
from torch.nn import functional as F 

torch.manual_seed(311)

class BaseLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batch = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        output = self.conv(x)
        output = self.batch(output)
        output = self.relu(output)
        return output

class BottleNeck(nn.Module):
    expend_dim = 4
    def __init__(self, in_channel, out_channel, stride=1):
        super(BottleNeck, self).__init__()
        self.layer1 = BaseLayer(in_channel=in_channel, out_channel=out_channel, kernel_size=1, stride=1, padding=1)
        self.layer2 = BaseLayer(in_channel=out_channel, out_channel=out_channel, kernel_size=3, stride=1, padding=1)
        self.layer3 = BaseLayer(in_channel=out_channel, out_channel=out_channel*self.expend_dim, kernel_size=1, stride=stride, padding=1)
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(x)
        output = self.layer3(x)
        return output
    
class Plain(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(Plain, self).__init__()
        self.layer1 = BaseLayer(in_channel=in_channel, out_channel=out_channel, kernel_size=3, stride=1, padding=1)
        self.layer2 = BaseLayer(in_channel=out_channel, out_channel=out_channel, kernel_size=3, stride=stride, padding=1)
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(x)
        return output
        
class ResNet(nn.Module):
    def __init__(self, Block, in_channel=3, num_layer=[2,2,2,2]):
        super(ResNet, self).__init__()
        self.block1 = BaseLayer(in_channel, out_channel=64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.block2 = self._make_layer(Block, in_ch=64, out_ch=128, num_layer=num_layer[0], stride=1)
        
    def _make_layer(self, Block, in_ch, out_ch, num_layer, stride):
        layers = []
        # if stride == 1:
        for num_ in range(num_layer-1):
            layers.append(Block(in_ch, out_ch))
            in_ch = out_ch
        layers.append(Block(in_ch, out_ch, stride=stride))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        output = self.block1(x)
        output = self.maxpool(output)
        output = self.block2(output)
        return output
    
if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    model = ResNet(Block=Plain)
    output = model(x)
    print(model)