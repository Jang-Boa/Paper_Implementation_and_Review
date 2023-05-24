import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

torch.manual_seed(311)

class DenseBlock(nn.Module):
    def __init__(self, in_feature, k):
        super(DenseBlock, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_feature)
        self.conv1 = nn.Conv2d(in_channels=in_feature, out_channels=k*4, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=k*4, out_channels=k, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x.clone()
        out = self.conv1(x)
        out = self.conv2(x)
        out = torch.cat([out, residual])
        return out

class TransitionLayer(nn.Module):
    def __init__(self, in_feature):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_feature)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=in_feature, out_channels=in_feature, kernel_size=1)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.avgpool(self.conv(self.relu(self.bn(x))))
        return out

class DenseNet(nn.Module):
    def __init__(self, in_feature=3, out_feature=1000, dense_block=[6, 12, 24, 16], k=32):
        super(DenseNet, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.dense_block = dense_block
        self.k = k
        self.bn = nn.BatchNorm2d(in_channel=self.in_feature)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=self.in_feature, out_channels=k*2, kernel_size=7, stride=2, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        for dense in dense_block:
            self. 
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=3), 
            nn.Linear(in_features=2048, out_features=self.out_feature)
        )
    def _make_layer(self):
        dense_block = dict()
        for dense in self.dense_block:
            for loop in range(dense):
                dense_block[f'dense_{loop+1}'] = DenseBlock(in_feature=self.k*2, out_feature=self.k)
                dense_block[f"transition_{loop+1}"] = TransitionLayer(in_feature=self.k)

        return nn.Sequential(**dense_block)

    
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.maxpool(out)
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    x = torch.randn(3, 299, 299)
    model = DenseNet()
    out = model(x)
    summary(model, (3, 299, 299))