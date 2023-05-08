import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

torch.manual_seed(311)

class InceptionBlock(nn.Module): 
    """ Inception Block
    23.05.08 패턴을 모르겠음
    """
    def __init__(self, in_features, layer1_feature, layer2_feature, layer3_feature, layer4_feature):
        super(InceptionBlock, self).__init__()
        self.in_features = in_features
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_features, out_channels=layer1_feature[0], kernel_size=1, stride=1, padding=0),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_features, out_channels=layer2_feature[0], kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=layer2_feature[0], out_channels=layer2_feature[1], kernel_size=3, stride=1, padding=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_features, out_channels=layer3_feature[0], kernel_size=1, stride=1, padding=1),
            nn.Conv2d(in_channels=layer3_feature[0], out_channels=layer3_feature[1], kernel_size=5, stride=1, padding=1)
        )
        self.proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=self.in_features, out_channels=layer4_feature[0], kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.proj(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out
    
class Auxiliary(nn.Module):
    def __init__(self, in_feature, out_feature=1000):
        super(Auxiliary, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3,padding=0)
        self.conv1 = nn.Conv2d(in_channels=self.in_feature, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=2048, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=self.out_feature)
        self.dropout = nn.Dropout2d(p=0.7)
        
    def forward(self, x):
        out = self.avgpool(x)
        out = self.conv1(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
        

class GoogLeNet(nn.Module):
    def __init__(self, in_features=3, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.num_classes = num_classes
        self.stem1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=64, kernel_size=7, stride=2, padding=3), 
            # nn.LocalResponseNorm(self.out_features),
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1),
            # nn.LocalResponseNorm(self.out_features*3/2), 
        )
        """ Pattern을 알면 코드를 줄일 수 있을 것 """
        self.inception3a = InceptionBlock(in_features=192, layer1_feature=[64], layer2_feature=[96, 128], layer3_feature=[16, 32], layer4_feature=[32])
        self.inception3b = InceptionBlock(in_features=256, layer1_feature=[128], layer2_feature=[128, 192], layer3_feature=[32, 96], layer4_feature=[64])
        self.inception4a = InceptionBlock(in_features=480, layer1_feature=[192], layer2_feature=[96, 208], layer3_feature=[16, 48], layer4_feature=[64])
        self.inception4b = InceptionBlock(in_features=512, layer1_feature=[160], layer2_feature=[112, 224], layer3_feature=[24, 64], layer4_feature=[64])
        self.inception4c = InceptionBlock(in_features=512, layer1_feature=[128], layer2_feature=[128, 256], layer3_feature=[24, 64], layer4_feature=[64])
        self.inception4d = InceptionBlock(in_features=512, layer1_feature=[112], layer2_feature=[144, 288], layer3_feature=[32, 64], layer4_feature=[64])
        self.inception4e = InceptionBlock(in_features=528, layer1_feature=[256], layer2_feature=[160, 320], layer3_feature=[32, 128], layer4_feature=[128])
        self.inception5a = InceptionBlock(in_features=832, layer1_feature=[256], layer2_feature=[160, 320], layer3_feature=[32, 128], layer4_feature=[128])
        self.inception5b = InceptionBlock(in_features=832, layer1_feature=[384], layer2_feature=[192, 384], layer3_feature=[48, 128], layer4_feature=[128])
        
        self.auxiliary1 = Auxiliary(in_feature=512, out_feature=self.num_classes)
        self.auxiliary2 = Auxiliary(in_feature=528, out_feature=self.num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(in_features=1024, out_features=self.num_classes)
        
    def forward(self, x):
        out = self.relu(self.stem1(x))
        out = self.maxpool(out)
        out = self.relu(self.stem2(out))
        out = self.maxpool(out)
        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.maxpool(out)
        out = self.inception4a(out)
        aux_output1 = self.auxiliary1(out)
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        aux_output2 = self.auxiliary2(out)
        out = self.inception4e(out)
        out = self.maxpool(out)
        out = self.inception5a(out)
        out = self.inception5b(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out) # Dropout의 순서는? Flatten 전 or 후?
        out = self.fc(out)
        return out, aux_output1, aux_output2
    

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = GoogLeNet(num_classes=10)
    output, aux_output1, aux_output2 = model(x)
    summary(model, (3, 224, 224))
    # print(model)
    print(output.shape, aux_output1.shape, aux_output2.shape)