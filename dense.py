import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from torchvision import models

model = models.densenet121()

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        model = models.densenet121()
        self.conv1 = model.features.conv0
        self.denseblock1 = model.features.denseblock1
        self.transition1 = model.features.transition1
        print(model)
    def forward(self,x):
        x = self.conv1(x)
        x = F.max_pool2d(x,2)
        print(x.size())
        x = self.denseblock1(x)
        print(x.size())
        pass

test_ = test()
test_.forward(torch.rand(1,3,256,256))
# summary(model, torch.randn((3, 224, 224)), device='cpu')