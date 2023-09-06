import torch 
from torch import nn 
from torchvision import models 

torch.manual_seed(311)

def set_parameter_requires_grad(model, freezing_layer_name=None, feature_extracting=False):
    """
    Fine-Tuning(False) or Feature Extracing(True)
    Flag for feature extracting
    When False, we finetune the whole model
    When True, we only update the reshaped layer params
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False # Freeze parameters
    else:
        if freezing_layer_name:
            cond = False
            for name, param in model.named_parameters():
                if freezing_layer_name == name:
                    cond = True # Freeze 
                param.requires_grad = cond
        else:
            for param in model.parameters():
                param.requires_grad = True # Trainable

def initialize_model(model_name, num_classes, layer_name=None, feature_extract=False, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet50":
        """ Resnet50
        input shape (224,224,3)
        """
        model_ft = models.resnet50(weights=use_pretrained)
        set_parameter_requires_grad(model_ft, freezing_layer_name=layer_name, feature_extracting=feature_extract)
        num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, num_classes),
                                    )
    
    elif model_name == "resnet101":
        """ Resnet101
        input shape (224,224,3)
        """
        model_ft = models.resnet101(weights=use_pretrained)
        set_parameter_requires_grad(model_ft, freezing_layer_name=layer_name, feature_extracting=feature_extract)
        num_ftrs = model_ft.fc.in_features
#         model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, num_classes),
                                    )
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft

class UNet_Custom(nn.Module):
    def __init__(self, num_classes=1, weight_path=None, num_features=[128, 256, 512, 1024]):
        super(UNet_Custom, self).__init__()
        model = initialize_model(model_name, num_classes, layer_name=layer_name, feature_extract=feature_extract, use_pretrained=use_pretrained)
        child_list = list(model.children()) # list(model.module.children())
        self.conv1 = child_list[0] # nn.Sequential(*list(model.module.children())[:1])
        self.bn1 = child_list[1]
        self.relu = child_list[2]
        self.maxpool = child_list[3]
        self.layer1 = child_list[4]
        self.layer2 = child_list[5]
        self.layer3 = child_list[6]
        self.layer4 = child_list[7]
        
        self.up4 = self.__upsample__(in_feature=num_features[3]*2, out_feature=num_features[2])
        self.up3 = self.__upsample__(in_feature=num_features[2]*3, out_feature=num_features[1])
        self.up2 = self.__upsample__(in_feature=num_features[1]*3, out_feature=num_features[0])
        self.up1 = self.__upsample__(in_feature=num_features[0]*3, out_feature=64)
        
        self.final_conv = nn.ConvTranspose2d(in_channels=64, out_channels=num_classes, kernel_size=2, stride=2)
        
        for m in self.modules(): # Using Kaiming initializer, initialize the weight of upsampling
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                
    def __upsample__(self, in_feature, out_feature):
        layers = []
        
        mid_feature = int(in_feature/2)
        for layer in range(2): 
            layers += [nn.Conv2d(in_channels=in_feature, out_channels=mid_feature, kernel_size=3, stride=1, padding=1), 
                       nn.BatchNorm2d(num_features=mid_feature), 
                       nn.ReLU()]
            in_feature = mid_feature
        layers += [nn.ConvTranspose2d(in_channels=mid_feature, out_channels=out_feature, kernel_size=2, stride=2)]
        
        for m in self.modules(): # Using Kaiming initializer, initialize the weight of upsampling
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight.data)
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_uniform_(m.weight.data)
        
        return nn.Sequential(*layers)        
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        u4 = self.up4(x4)
        u4 = torch.cat((u4, x3), dim=1)
        
        u3 = self.up3(u4)
        u3 = torch.cat((u3, x2), dim=1)
        
        u2 = self.up2(u3)
        u2 = torch.cat((u2, x1), dim=1)
        
        u1 = self.up1(u2)
        
        output = self.final_conv(u1)
        return output
    
if __name__ == '__main__':
    x = torch.randn(2, 3, 512, 512)
    model_name = 'resnet50'
    layer_name = None
    feature_extract = False
    use_pretrained = None
    model = UNet_Custom(2, weight_path=None)
    print(model)
    output = model(x)
    print(output.shape)