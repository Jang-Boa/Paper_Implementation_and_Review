import torch 
from torch import nn 
 
torch.manual_seed(42)

class FCN(nn.Module):
    """ VGG16-based Net """
    def __init__(self) -> None:
        super().__init__(FCN)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=False),
            )
    def _make_block(self, x, num_layer):
        layers = []
        for layer in range(num_layer):
            info = nn.
            layers.append()

if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    model = FCN()
    output = model(x)