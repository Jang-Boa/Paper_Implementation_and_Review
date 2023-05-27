import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

torch.manual_seed(311)

class InceptionA(nn.Module):
    def __init__(self, in_features, out_features):
        super(InceptionA, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features[0], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_features[0], out_channels=out_features[0], kernel_size=3, stride=1, padding=0),
            nn.Conv2d(in_channels=out_features[0], out_channels=out_features[0], kernel_size=3, stride=2, padding=1),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features[1], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_features[1], out_channels=out_features[1], kernel_size=3, stride=2, padding=1),
        )
        self.layer3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=in_features, out_channels=out_features[2], kernel_size=1, stride=1, padding=0),
        )
        self.layer4 = nn.Conv2d(in_channels=in_features, out_channels=out_features[3], kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

class InceptionB(nn.Module):
    def __init__(self, in_features, out_features, n=7):
        super(InceptionB, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features[0], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_features[0], out_channels=out_features[0], kernel_size=(1, n), stride=1, padding=0),
            nn.Conv2d(in_channels=out_features[0], out_channels=out_features[0], kernel_size=(n, 1), stride=1, padding=0),
            nn.Conv2d(in_channels=out_features[0], out_channels=out_features[0], kernel_size=(1, n), stride=1, padding=0),
            nn.Conv2d(in_channels=out_features[0], out_channels=out_features[0], kernel_size=(n, 1), stride=1, padding=0),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features[1], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_features[1], out_channels=out_features[1], kernel_size=(1, n), stride=1, padding=0),
            nn.Conv2d(in_channels=out_features[1], out_channels=out_features[1], kernel_size=(n, 1), stride=1, padding=0),
        )
        self.layer3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=in_features, out_channels=out_features[2], kernel_size=1, stride=1, padding=0),
        )
        self.layer4 = nn.Conv2d(in_channels=in_features, out_channels=out_features[3], kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out
    
class InceptionC(nn.Module):
    def __init__(self, in_features, out_features):
        super(InceptionC, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features[0], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=out_features[0], out_channels=out_features[0], kernel_size=3, stride=1, padding=0),
        )
        self.layer1a = nn.Conv2d(in_channels=out_features[0], out_channels=out_features[0], kernel_size=(1, 3), stride=2, padding=0)
        self.layer1b = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features[1], kernel_size=1, stride=1, padding=0),
        )
        self.layer2a = nn.Conv2d(in_channels=out_features[1], out_channels=out_features[1], kernel_size=(1, 3), stride=2, padding=0)
        self.layer2b = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.layer3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1),
            nn.Conv2d(in_channels=in_features, out_channels=out_features[2], kernel_size=1, stride=1, padding=0),
        )
        self.layer4 = nn.Conv2d(in_channels=in_features, out_channels=out_features[3], kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.layer1(x)
        out1a = self.layer1a(out1)
        out1b = self.layer1b(out1)
        out2 = self.layer2(x)
        out2a = self.layer1a(out2)
        out2b = self.layer1b(out2)
        out3 = self.layer3(x)
        out4 = self.layer4(x)
        out = torch.cat([out1a, out1b, out2a, out2, out3, out4], dim=1)
        return out

class GridReduction(nn.Module):
    def __init__(self, in_features, out_features):
        super(GridReduction, self).__init__()
        self.grid1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=(3, 3), stride=2, padding=0)
        self.grid2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

    def forward(self, x):
        out1 = self.grid1(x)
        out2 = self.grid2(x)
        out = torch.cat([out1, out2], dim=1)
        return out
    
class Auxiliary(nn.Module):
    def __init__(self, in_feature, out_feature=1000):
        super(Auxiliary, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3,padding=0)
        self.conv1 = nn.Conv2d(in_channels=self.in_feature, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.dropout = nn.Dropout2d(p=0.7)
        self.fc2 = nn.Linear(in_features=1024, out_features=self.out_feature)
        
    def forward(self, x):
        out = self.avgpool(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class InceptionV2(nn.Module):
    def __init__(self, in_features=3, num_classes=1000):
        super(InceptionV2, self).__init__()
        self.num_classes = num_classes
        self.stem1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0), 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),# Conv Padded (?)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.stem2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, stride=1, padding=0),
            nn.Conv2d(in_channels=80, out_channels=192, kernel_size=3, stride=2, padding=0),
            nn.Conv2d(in_channels=192, out_channels=288, kernel_size=3, stride=1, padding=1),
        )
        self.inceptionA1 = InceptionA(in_features=288, out_features=768) # out_features ?
        self.inceptionA2 = InceptionA(in_features=288, out_features=)
        self.inceptionA3 = InceptionA(in_features=288, out_features=)
        self.ReductionA = GridReduction()
        self.inceptionB1 = InceptionB(in_features=288, out_features=)
        self.inceptionB2 = InceptionB(in_features=288, out_features=)
        self.inceptionB3 = InceptionB(in_features=288, out_features=)
        self.inceptionB4 = InceptionB(in_features=288, out_features=)
        self.inceptionB5 = InceptionB(in_features=288, out_features=)
        self.ReductionB = GridReduction()
        self.inceptionC1 = InceptionC(in_features=1280, out_features=)
        self.inceptionC2 = InceptionC(in_features=288, out_features=2048)

        self.avgpool = nn.AdaptiveAvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(in_features=2048, out_features=self.num_classes)

        self.auxilary = Auxiliary(in_feature=768, out_feature=self.num_classes)
        
    def forward(self, x):
        out = self.relu(self.stem1(x))
        out = self.maxpool(out)
        out = self.relu(self.stem2(out))
        out = self.inceptionA1(out)
        # out = self.fc(out)

        return out
    
if __name__ == '__main__':
    x = torch.randn(1, 3, 299, 299)
    model = InceptionV2()
    out = model(x)
    summary(model, (3, 299, 299))