import torch 
import torch.nn as nn

torch.manual_seed(42)

class UNet(nn.Module):
    def __init__(self, num_channel=3, num_features=[64, 128, 256, 512, 1024], out_feature=2):
        super(UNet, self).__init__()
        self.down1 = self.__downsample__(in_feature=num_channel, out_feature=num_features[0])
        self.down2 = self.__downsample__(in_feature=num_features[0], out_feature=num_features[1])
        self.down3 = self.__downsample__(in_feature=num_features[1], out_feature=num_features[2])
        self.down4 = self.__downsample__(in_feature=num_features[2], out_feature=num_features[3])
        self.down5 = self.__downsample__(in_feature=num_features[3], out_feature=num_features[4])
        self.up5 = self.__upsample__(in_feature=num_features[4], out_feature=num_features[3])
        self.up4 = self.__upsample__(in_feature=num_features[3]*2, out_feature=num_features[2])
        self.up3 = self.__upsample__(in_feature=num_features[2]*2, out_feature=num_features[1])
        self.up2 = self.__upsample__(in_feature=num_features[1]*2, out_feature=num_features[0])
        self.up1 = self.__upsample__(in_feature=num_features[0]*2, out_feature=out_feature)

    def __downsample__(self, in_feature, out_feature):
        layers = []
        if in_feature <= 3:
            pass
        else: 
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        for layer in range(2):
            layers += [nn.Conv2d(in_channels=in_feature, out_channels=out_feature, kernel_size=3, stride=1, padding=0),
                       nn.BatchNorm2d(num_features=out_feature), 
                       nn.ReLU()]
            in_feature = out_feature
        
        return nn.Sequential(*layers)
      
    def __upsample__(self, in_feature, out_feature):
        layers = []
        if in_feature==(out_feature*2):
            layers += [nn.ConvTranspose2d(in_channels=in_feature, out_channels=out_feature, kernel_size=2, stride=2)]
        elif out_feature < 64:
            mid_feature = int(in_feature/2)
            for layer in range(2):
                layers += [nn.Conv2d(in_channels=in_feature, out_channels=mid_feature, kernel_size=3, stride=1, padding=0), 
                           nn.BatchNorm2d(num_features=mid_feature), 
                           nn.ReLU()]
                in_feature = mid_feature
            layers += [nn.Conv2d(in_channels=mid_feature, out_channels=out_feature, kernel_size=1, stride=1, padding=0)]
        else:
            mid_feature = int(in_feature/2)
            for layer in range(2):
                layers += [nn.Conv2d(in_channels=in_feature, out_channels=mid_feature, kernel_size=3, stride=1, padding=0), 
                           nn.BatchNorm2d(num_features=mid_feature), 
                           nn.ReLU()]
                in_feature = mid_feature
            layers += [nn.ConvTranspose2d(in_channels=mid_feature, out_channels=out_feature, kernel_size=2, stride=2)]
            
        return nn.Sequential(*layers)
    
    def crop_and_resize(self, down, up):
        crop = int((down.size()[2] - up.size()[2])/2)
        total = down.size()[2]
        down = down[:,:,crop:total-crop, crop:total-crop]
        return down


    def forward(self, x):
        d_out1 = self.down1(x)
        d_out2 = self.down2(d_out1)
        d_out3 = self.down3(d_out2)
        d_out4 = self.down4(d_out3)
        d_out5 = self.down5(d_out4)

        u_out5 = self.up5(d_out5)
        d_out4 = self.crop_and_resize(down=d_out4, up=u_out5)
        u_out4 = torch.cat((u_out5, d_out4), dim=1)

        u_out4 = self.up4(u_out4)
        d_out3 = self.crop_and_resize(down=d_out3, up=u_out4)
        u_out3 = torch.cat((u_out4, d_out3), dim=1)
        
        u_out3 = self.up3(u_out3)
        d_out2 = self.crop_and_resize(down=d_out2, up=u_out3)
        u_out2 = torch.cat((u_out3, d_out2), dim=1)

        u_out2 = self.up2(u_out2)
        d_out1 = self.crop_and_resize(down=d_out1, up=u_out2)
        u_out1 = torch.cat((u_out2, d_out1), dim=1)

        output = self.up1(u_out1)
        return output

if __name__ == "__main__":
    x = torch.rand(2, 3, 572, 572)
    model = UNet()
    print(model)
    output = model(x)
    print(output.size())