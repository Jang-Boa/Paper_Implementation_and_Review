import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

torch.manual_seed(311)

# class DenseBlock(nn.Module):
#     """DenseNet-BC structure with 4 dense blocks on 224 x 224 images """
#     def __init__(self, in_feature, k):
#         super(DenseBlock, self).__init__()
#         self.bn = nn.BatchNorm2d(num_features=in_feature)
#         self.conv1 = nn.Conv2d(in_channels=in_feature, out_channels=k*4, kernel_size=1, stride=1)
#         self.conv2 = nn.Conv2d(in_channels=k*4, out_channels=k, kernel_size=3, stride=2, padding=1)
#         self.relu = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         residual = x.clone()
#         out = self.conv1(x)
#         out = self.conv2(x)
#         out = torch.cat([out, residual])
#         return out

class BottleneckLayer(nn.Module):
    def __init__(self, in_feature, k):
        super(BottleneckLayer, self).__init__()
        """ BN-ReLU-Conv1-BN-ReLU-Conv3
        Each layer only produces k output feature maps """
        self.inter_feature = 4 * k
        self.bn1 = nn.BatchNorm2d(in_feature)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_feature, out_channels=self.inter_feature, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=self.inter_feature)
        self.conv2 = nn.Conv2d(in_channels=self.inter_feature, out_channels=k, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_ = x.clone()
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = torch.cat([x_, out], 1) # original concatenate new
        return out


class TransitionLayer(nn.Module):
    """ transition layers consist of a batch normalization layer and an 1 x 1 convolutional layer followed by a 2x2 average pooling layer 
    Reduce feature map size and number of channels """
    def __init__(self, in_feature, out_feature):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_feature)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=1, stride=1, padding=0, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.avgpool(self.conv(self.relu(self.bn(x))))
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_feature, k):
        super(DenseBlock, self).__init__()
        self.inner_channel = 4 * k
        self.bn1 = nn.BatchNorm2d(num_features=in_feature)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_feature, out_channels=self.inner_channel, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=self.inner_channel)
        self.conv2 = nn.Conv2d(in_channels=self.inner_channel, out_channels=k, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        return out

class DenseNet(nn.Module):
    def __init__(self, in_feature=3, num_classes=1000, dense_block=[6, 12, 24, 16], k=32, compression=0.5):
        super(DenseNet, self).__init__()
        self.in_feature = in_feature
        self.dense_block = dense_block
        self.in_plane = 2 * k # 1st channel 
        self.conv = nn.Conv2d(in_channels=self.in_feature, out_channels=self.in_plane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(num_features=self.in_plane)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 1st Dense Block followed by Transition Layer
        self.denseblock1 = self._make_dense_block(self.in_plane, k, dense_block[0])
        self.in_plane = self.in_plane*2
        self.transition1 = TransitionLayer(self.in_plane*2, self.in_plane)
        self.denseblock2 = self._make_dense_block(self.in_plane, k, dense_block[1])
        self.in_plane = self.in_plane*2
        self.transition2 = TransitionLayer(self.in_plane*2, self.in_plane)
        self.denseblock3 = self._make_dense_block(self.in_plane, k, dense_block[2])
        self.in_plane = self.in_plane*2
        self.transition3 = TransitionLayer(self.in_plane*2, self.in_plane)
        self.denseblock4 = self._make_dense_block(self.in_plane, k, dense_block[3])

        # self.classifier = nn.Sequential(
        #     nn.AvgPool2d(kernel_size=7, stride=3), 
        #     nn.Linear(in_features=2048, out_features=num_classes)
        # )
        
    def _make_dense_block(self, in_feature, k, dense):
        layers = []
        for loop in range(dense):
            # print(loop)
            layers.append(BottleneckLayer(in_feature=in_feature, k=k))
            # dense_block[f"transition_{loop}"] = TransitionLayer(in_feature=in_feature, k=k)
            in_feature += k
        return nn.Sequential(*layers)

    
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        print(out.shape)
        out = self.maxpool(out)
        print(out.shape)
        out = self.denseblock1(out)
        print(out.shape)
        out = self.transition1(out)
        print(out.shape)
        out = self.denseblock2(out)
        print(out.shape)
        out = self.transition2(out)
        print(out.shape)
        out = self.denseblock3(out)
        print(out.shape)
        out = self.transition3(out)
        print(out.shape)
        out = self.denseblock4(out)
        print(out.shape)
        # out = self.classifier(out)
        return out

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = DenseNet()
    out = model(x)
    print(model)
    # summary(model, (3, 224, 224))