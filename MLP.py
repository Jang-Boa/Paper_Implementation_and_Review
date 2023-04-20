import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
	def __init__(self,inputs):
		super(MLP, self).__init__()
		self.layer = nn.Linear(inputs, 1)
		self.activation = nn.Sigmoid()
	def forward(self, x):
		x = F.ReLU(self.layer(x))
		x = self.activation(x)
		return x
