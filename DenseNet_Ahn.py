import torch
import torch.nn as nn
from collections import OrderedDict
from torchsummary import summary

torch.manual_seed(311)
""" Rewrite 23.05.31 """

class DenseLayer(nn.Module):
    def __init__(self, in_feature, growth_rate):
        super().__init__()
        inter_channel = 4*growth_rate
        self.bn1 = nn.BatchNorm2d(in_feature)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_feature, out_channels=inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channel)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=inter_channel, out_channels=growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return out # torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_feature, num_layer, growth_rate):
        super().__init__()
        for idx in range(num_layer):
            input_feature = in_feature + idx * growth_rate
            layer = DenseLayer(input_feature, growth_rate)
            self.add_module("denselayer%d"%(idx+1), layer)
            
    def forward(self, init_feature):
        features = init_feature.clone()
        for name, layer in self.named_children():
            new_features = layer(features)
            features = torch.cat([features, new_features], 1)
            # print(name, features.size())
        print(features.shape)
        return features
    
class TransitionBlock(nn.Module):
    def __init__(self, in_feature, compression):
        super(TransitionBlock, self).__init__()
        out_feature = int(in_feature*compression)
        self.bn = nn.BatchNorm2d(in_feature)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=1, stride=1, padding=0, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        
    def forward(self, x):
        out = self.pool(self.conv(self.relu(self.bn(x))))
        return out
    
class DenseNet(nn.Module):
    def __init__(self, in_feature=3, num_classes=1000, num_dense_block=[6, 12, 24, 16], growth_rate=32, compression=0.5):
        super(DenseNet, self).__init__()
        initial_feature = 2*growth_rate
        self.encoder = nn.Sequential(
            OrderedDict([
                ("conv0", nn.Conv2d(in_channels=in_feature, out_channels=initial_feature, kernel_size=7, stride=2, padding=3, bias=False)), 
                ("bn0", nn.BatchNorm2d(num_features=initial_feature)),
                ("relu0", nn.ReLU(inplace=True)),
                ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ])
        )
        
        for idx, num_layer in enumerate(num_dense_block):
            self.encoder.add_module("denseblock%d"%(idx+1), DenseBlock(initial_feature, num_layer, growth_rate))
            initial_feature = initial_feature + num_layer * growth_rate
            if idx != len(num_dense_block)-1:
                self.encoder.add_module("transition%d"%(idx+1), TransitionBlock(initial_feature, compression))
                initial_feature = initial_feature//2
        
        self.bn1 = nn.BatchNorm2d(initial_feature)
        self.avgpool1 = nn.AvgPool2d(kernel_size=7, stride=3)
        self.classifier = nn.Linear(in_features=initial_feature, out_features=num_classes, bias=True)
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.avgpool1(self.bn1(out))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
    
if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224)
    model = DenseNet(num_classes=10)# , num_dense_block=[6, 12, 32, 32], growth_rate=32)
    out = model(x)
    print(out)
    # print(model)
    print(out.shape)
    # print(summary(model, (3, 224, 224)))