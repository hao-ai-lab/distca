import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    def __init__(self, dim, tp):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4 // tp)
        self.fc2 = nn.Linear(dim * 4 // tp, dim)
        self.dim_shape = (dim, dim * 4 // tp)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

mlp = MLP(1024, 1)
mlp.compile()

x = torch.randn(100)
y = mlp(x)

print(y)