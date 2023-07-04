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
        self.head.add_module("block1", self._make_block(layer_dict['block1']))
        # self.fc = nn.Sequential(
        #     nn.Linear(in_features=4096, out_features=1024), 
        #     nn.ReLU(),  
        #     nn.Linear(in_features=1024, out_features=num_classes)
        # )

    def _make_block(self, num_layer):
        block = []
        for num_channel in num_layer:
            print(num_channel)
            block += [nn.Conv2d(in_channels=num_channel, out_channels=num_channel, kernel_size=3, stride=1, padding=1), 
                     nn.BatchNorm2d(num_features=num_channel), 
                     nn.ReLU()]
            if num_channel == 'MP': 
                block += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*block)
    
    def forward(self, x):
        output = self.head(x)
        print(output.shape)
        output = torch.flatten(output, 1)
        print(output.shape)
        # output = self.fc(output)
        return output

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    layer_dict = {"block1": [64, 64, 'MP'], 
                  "block2": [128, 128, 'MP'], 
                  "block3": [256, 256, 256, 'MP'],
                  "block4": [512, 512, 512, 'MP'],
                  "block5": [512, 512, 512]}
    print(layer_dict['block1'])
    model = FCN(layer_dict, num_classes=10)
    print(model)
    # output = model(x)