import torch 
from torch import nn 
 
torch.manual_seed(42)

class FCN(nn.Module):
    """ VGG16-based Net """
    def __init__(self, layer_dict, num_classes=10):
        super(FCN, self).__init__()
        self.layer_dict = layer_dict
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=False),
            )
        self.downsample1 = self._make_downsample(64, layer_dict['block1'])
        self.downsample2 = self._make_downsample(layer_dict['block1'][0], layer_dict['block2'])
        self.downsample3 = self._make_downsample(layer_dict['block2'][0], layer_dict['block3'])
        self.downsample4 = self._make_downsample(layer_dict['block3'][0], layer_dict['block4'])
        self.downsample5 = self._make_downsample(layer_dict['block4'][0], layer_dict['block5'])

        self.fc = nn.Sequential(
            nn.Linear(in_features=1*1*512, out_features=1024), 
            nn.ReLU(),  
            nn.Linear(in_features=1024, out_features=num_classes)
        )
        self.upsample5 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=32, stride=1)
        # self.upsample4 = nn.ConvTranspose2d(in_channels=512, out_channels=1, kernel_size=32, stride=16)

    def _make_downsample(self, in_feature, num_layer):
        block = []
        for num_channel in num_layer:
            if num_channel == 'MP': 
                block += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_feature = num_channel
                block += [nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=3, stride=1, padding=1), 
                        nn.BatchNorm2d(num_features=out_feature), 
                        nn.ReLU()]
                in_feature = num_channel
            
        return nn.Sequential(*block)
    
    
    def forward(self, x):
        output = self.head(x)
        output = self.downsample1(output)
        output = self.downsample2(output)
        output = self.downsample3(output)
        output = self.downsample4(output)
        output5 = self.downsample5(output)
        output = self.downsample5(output)
        
        print(output.shape)
        print(output5.shape)
        
        output = self.upsample5(output)
        print(output.shape)
        # output = output + output5
        
        # output = self.upsample4(output)
        # print(output.shape)
        # output = torch.flatten(output, 1)
        # output = self.fc(output)
        return output

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    layer_dict = {"block1": [64, 64, 'MP'], 
                  "block2": [128, 128, 'MP'], 
                  "block3": [256, 256, 256, 'MP'],
                  "block4": [512, 512, 512, 'MP'],
                  "block5": [512, 512, 512, 'MP']}
    model = FCN(layer_dict, num_classes=10)
    # print(model)
    # print(x.shape)
    output = model(x)
    # print(output.shape)