import torch 
import torch.nn as nn
from collections import OrderedDict

torch.manual_seed(311)

""" 안영빈 선생님 코드 """

crf = {
    'layer1': [3,64,2,True],
    'layer2': [64,128,2,True],
    'layer3': [128,256,3,True],
    'layer4': [256,512,4,True],
    'layer5': [512,512,4,True]
}


class VGGNet(nn.Module):
    def BasicBlock(self, init, out, kernel_size=3, stride=1, padding=1):
        basic_block = nn.Sequential(
            nn.Conv2d(init, out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU())
        return basic_block
            
    def Make_layer(self,init,out,iteration=1,maxpool=True):
        layers = [self.BasicBlock(init,out)]     
        for _ in range(1,iteration):
            layers.append(self.BasicBlock(out,out))
        if maxpool:
            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
        return nn.Sequential(*layers)
    
    def Make_encoder(self,crf):
        layer1 = self.Make_layer(*crf['layer1'])
        layer2 = self.Make_layer(*crf['layer2'])
        layer3 = self.Make_layer(*crf['layer3'])
        layer4 = self.Make_layer(*crf['layer4'])
        layer5 = self.Make_layer(*crf['layer5'])
        
        encoder = nn.Sequential(OrderedDict([
            ('layer1',layer1),
            ('layer2',layer2),
            ('layer3',layer3),
            ('layer4',layer4),
            ('layer5',layer5)
        ]))
        return encoder
    
    def __init__(self,crf, num_classes=10):
        super(VGGNet, self).__init__()
            
        self.encoder = self.Make_encoder(crf)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.classifier(out)
        return out
    
if __name__ == '__main__':
    x = torch.rand(3, 3, 224, 224) 
    model = VGGNet(crf)
    print(model)
    output = model(x)
    print('-'*50)
    print(output)