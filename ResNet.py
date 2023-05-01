import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(311)

class BottleNeck(nn.Module):
    """ BottleNeck의 핵심은 1x1 Conv. ; Three-laer bottleneck block """
    expansion = 4
    def __init__(self, in_features, out_features, downsample=None, residual=True, stride=1):
        super(BottleNeck, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0, bias=False) # 1x1 conv, reducing dimension
        self.bn_1 = nn.BatchNorm2d(out_features)

        self.conv_2 = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=stride, padding=1, bias=False) # 3x3 conv
        self.bn_2 = nn.BatchNorm2d(out_features)
    
        self.conv_3 = nn.Conv2d(in_channels=out_features, out_channels=out_features * self.expansion, kernel_size=1, stride=1, padding=0, bias=False) # 1x1 conv, increasing dimension; 
        self.bn_3 = nn.BatchNorm2d(out_features * self.expansion)
        
        self.downsample = downsample
        self.residual = residual
        self.stride=stride
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x.clone() # copy tensor 
        out = self.relu(self.bn_1(self.conv_1(x))) 
        out = self.relu(self.bn_2(self.conv_2(out)))
        out = self.bn_3(self.conv_3(out)) # the output of each 3*3 layer, after BN and before other nonlinearity (ReLU/addition)
        if self.downsample is not None: # downsample 
            identity = self.downsample(identity)
        if self.residual is True: 
            out += identity # shortcut connection simply perform identity mapping
        out = self.relu(out)
        return out

class Standard(nn.Module):
    """ Standarc Block은 3x3 conv. 2-layer block로 구성 
    """
    expansion = 1 # global 
    def __init__(self, in_features, out_features, downsample=None, residual=True, stride=1):
        super(Standard, self).__init__()
        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias = False호 설정
        self.conv_1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=stride, padding=1, bias=False) # stride -> 
        self.bn_1 = nn.BatchNorm2d(out_features)
        
        self.conv_2 = nn.Conv2d(in_channels=out_features, out_channels=out_features*self.expansion , kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn_2 = nn.BatchNorm2d(out_features*self.expansion)
        
        self.downsample = downsample
        self.residual = residual
        self.stride=stride
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x.clone()
        out = self.relu(self.bn_1(self.conv_1(x)))
        out = self.bn_2(self.conv_2(out))
        if self.downsample is not None: # DownSampling의 진행 여부
            identity = self.downsample(identity)
        if self.residual is True: # Residual Block의 사용 여부 
            out += identity # Shortcut connection : Identity Mapping (sufficient for address the degration problem)
        
        out = self.relu(out)
        return out

class ResNet(nn.Module):# 34-layer plain
    def __init__(self, Block, block_num, channel_size, num_channels=3, num_classes=1000, residual_block=True):
        super(ResNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode='zeros'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_features = channel_size[0]
        self.residual_block = residual_block
        self.layer2 = self._mask_layer(Block, num_layer=block_num[0], channel=channel_size[0])
        self.layer3 = self._mask_layer(Block, num_layer=block_num[1], channel=channel_size[1], stride=2)
        self.layer4 = self._mask_layer(Block, num_layer=block_num[2], channel=channel_size[2], stride=2)
        self.layer5 = self._mask_layer(Block, num_layer=block_num[3], channel=channel_size[3], stride=2)
        self.num_classes = num_classes

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channel_size[3]*Block.expansion, self.num_classes), 
            # nn.Softmax(dim=1),
        )
        
    def _mask_layer(self, Block, num_layer, channel, stride=1):
        downsample = None
        layers = []
        if stride != 1 or self.in_features != channel*Block.expansion: 
            # projection mapping using 1x1 conv
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.in_features, out_channels=channel*Block.expansion, kernel_size=1, stride=stride), # Projection shortcuts done by 1*1 conv.
                nn.BatchNorm2d(channel*Block.expansion)
            )
        layers.append(Block(self.in_features, channel, downsample=downsample, residual=self.residual_block, stride=stride))
        self.in_features = channel*Block.expansion
        for loop in range(num_layer-1):
            layers.append(Block(self.in_features, channel, residual=self.residual_block))
        
        return nn.Sequential(*layers)
        
    
    def forward(self, x):
        out = self.layer1(x)
        print(out.shape)
        out = self.maxpool(out)
        out = self.layer2(out)
        print(out.shape)
        out = self.layer3(out)
        print(out.shape)
        out = self.layer4(out)
        print(out.shape)
        out = self.layer5(out)
        print(out.shape)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = F.softmax(self.fc(out), dim=1)
        return out

if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)
    model = ResNet(Block=BottleNeck, block_num=[3, 4, 6, 3], channel_size=[64, 128, 256, 512], num_channels=3, num_classes=2, residual_block=True)
    print(model)
    output = model(x)
    print('-'*10)
    print(output)