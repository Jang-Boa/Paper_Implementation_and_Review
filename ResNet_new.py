import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

torch.manual_seed(311)

class BottleNeck(nn.Module):
    """ BottleNeck의 핵심은 1x1 Conv. ; Three-laer bottleneck block """
    expansion = 4
    def __init__(self, in_features, out_features, downsample=None, residual=True, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0) # 1x1 conv
        self.bn1 = nn.BatchNorm2d(out_features) # nn module 

        self.conv2 = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=stride, padding=1, bias=False) # 3x3 conv
        self.bn2 = nn.BatchNorm2d(out_features)
    
        self.conv3 = nn.Conv2d(in_channels=out_features, out_channels=out_features * self.expansion, kernel_size=1, stride=1, padding=0, bias=False) # 1x1 conv, increasing dimension; 
        self.bn3 = nn.BatchNorm2d(out_features * self.expansion)
        
        self.downsample = downsample
        self.residual = residual
        self.stride=stride
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x.clone() # copy tensor 
        out = self.relu(self.bn1(self.conv1(x))) 
        out = self.relu(self.bn2(self.conv2(out)))# the output of each 3*3 layer, after BN and before other nonlinearity (ReLU/addition)
        out = self.bn3(self.conv3(out))
        out = self.relu(out)
        if self.downsample is not None: # downsample 
            identity = self.downsample(identity)
        if self.residual is True: # gradient vanishing 방지를 위함
            out += identity # shortcut connection simply perform identity mapping
        
        return out

class Standard(nn.Module):
    """ Standarc Block은 3x3 conv. 2-layer block로 구성 
    """
    expansion = 1 # global 
    def __init__(self, in_features, out_features, downsample=None, residual=True, stride=1):
        super(Standard, self).__init__()
        # BatchNorm에 bias가 포함되어 있으므로, conv2d는 bias = False호 설정
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=stride, padding=1, bias=False) # stride -> 
        self.bn1 = nn.BatchNorm2d(out_features)
        
        self.conv2 = nn.Conv2d(in_channels=out_features, out_channels=out_features*self.expansion , kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(out_features*self.expansion)
        
        self.downsample = downsample
        self.residual = residual
        self.stride=stride
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x.clone()
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: # DownSampling의 진행 여부
            identity = self.downsample(identity)
        if self.residual is True: # Residual Block의 사용 여부 
            out += identity # Shortcut connection : Identity Mapping (sufficient for address the degration problem)
        out = self.relu(out)
        return out

class ResNetEncoder(nn.Module):# 34-layer plain
    def __init__(self, Block, block_num, channel_size=[64, 128, 256, 512], num_channels=3, residual_block=True):
        super(ResNetEncoder, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_features = channel_size[0]
        self.residual_block = residual_block
        self.layer1 = self._mask_layer(Block, num_layer=block_num[0], channel=channel_size[0])
        self.layer2 = self._mask_layer(Block, num_layer=block_num[1], channel=channel_size[1], stride=2)
        self.layer3 = self._mask_layer(Block, num_layer=block_num[2], channel=channel_size[2], stride=2)
        self.layer4 = self._mask_layer(Block, num_layer=block_num[3], channel=channel_size[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out

def resnet18(**kwargs):
    return ResNetEncoder(Block=Standard, block_num=[2, 2, 2, 2], **kwargs)

def resnet34(**kwargs):
    return ResNetEncoder(Block=Standard, block_num=[3, 4, 6, 3], **kwargs)

def resnet50(**kwargs):
    return ResNetEncoder(Block=BottleNeck, block_num=[3, 4, 6, 3], **kwargs)

def resnet101(**kwargs):
    return ResNetEncoder(Block=BottleNeck, block_num=[3, 4, 23, 3], **kwargs)

def resnet152(**kwargs):
    return ResNetEncoder(Block=BottleNeck, block_num=[3, 8, 36, 3], **kwargs)

model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    'resnet152': [resnet152, 2048],
}

class ResNet(nn.Module):
    def __init__(self, name='resnet50', head='mlp', num_classes=1000):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        model_func, dim_in = model_dict[name]
        self.encoder = model_func()
        if head=='linear':
            self.head = nn.Linear(dim_in, self.num_classes), 
        elif head=='mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in), 
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, self.num_classes)
            )
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.head(out)
        return out

if __name__ == '__main__':
    x = torch.rand(2, 3, 224, 224)
    model = ResNet(name='resnet18', head='mlp', num_classes=2)
    print(model)
    # summary(model, (3, 224, 224))
    output = model(x)
    print(F.softmax(output))