import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(42) # Fix random seed

class MLP(nn.Module):
	def __init__(self):
		super(MLP, self).__init__()
		self.layer1 = nn.Linear(in_features=12, out_features=6)
		self.layer2 = nn.Linear(in_features=6, out_features=1)
		self.activation = nn.Sigmoid()
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = self.activation(x)
		return x


if __name__ == '__main__':
	model = MLP()
	print("     <<< Model Architecture >>>     ")
	print(model)
	print('-'*10)
	x = torch.rand(12)
	output = model(x)
	print(f"Results: {output}")
	print("Finish!")