import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(311)

class SEBlock(nn.Module):
    def __init__(self):
        super(SEBlock, self).__init__()
        