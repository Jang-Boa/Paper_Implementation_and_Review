import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

torch.manual_seed(311)

class InceptionV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV2, self).__init__()
        x
    