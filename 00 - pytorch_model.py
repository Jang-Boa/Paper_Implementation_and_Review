import torch
from torch import nn 

torch.manual_seed(311) # Fix Tensor Random Seed
### Pytorch로 시작하는 딥러닝
### https://wikidocs.net/book/2788
### Pytorch Docs: https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

class BaseModel(nn.Module): # torch.nn.Module을 상속받은 파이썬 클래스
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1) # 단순 선형 회귀일 경우, input_dim=1, output_dim=1

    def forward(self, x):
        return self.linear(x)
    
if __name__== '__main__':
    x = torch.rand(1) # Randomly create new tensor
    model = BaseModel() 
    print(model)
    print(f"\nResults: {model(x)}")