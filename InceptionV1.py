import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(311)

class InceptionBlock(nn.Module):
    def __init__(self, in_features):
        super(InceptionBlock, self).__init__()
        self.in_features = in_features
        self.inception1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_features, out_channels=64, kernel_size=1, stride=1, padding=0),
        )
        self.inception2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_features, out_channels=96, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1)
        )
        self.inception3 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_features, out_channels=16, kernel_size=1, stride=1, padding=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1)
        )
        self.proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=self.in_features, out_channels=32, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        out1 = self.inception1(x)
        print(out1.shape)
        out2 = self.inception2(x)
        print(out2.shape)
        out3 = self.inception3(x)
        print(out3.shape)
        out4 = self.proj(x)
        print(out4.shape)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class GoogLeNet(nn.Module):
    def __init__(self, in_features=3, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.stem1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3), 
            # nn.LocalResponseNorm(self.out_features),
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            # nn.LocalResponseNorm(self.out_features*3/2), 
        )
        self.inception_block1 = InceptionBlock(in_features=192)
        self.inception_block2 = 
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)
        
    def forward(self, x):
        out = self.relu(self.stem1(x))
        print(out.shape)
        out = self.maxpool(out)
        print(out.shape)
        out = self.relu(self.stem2(out))
        print(out.shape)
        out = self.inception(out)
        print(out.shape)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        print(out.shape)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
    

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = GoogLeNet()
    output = model(x)
    print(model)