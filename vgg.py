import torch
from torch import nn as nn


# VGG16-based Model
class VGG(nn.Module):
  def __init__(self, input_dim=100*2, output_dims=(1, 20, 20)):
    super(VGG, self).__init__()

    self.input_dim = input_dim
    self.output_dims = output_dims
    int_dim = 100
    output_dim = int(torch.prod(torch.Tensor(output_dims), 0).item())

    self.net = nn.Sequential(
        nn.Linear(input_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),

        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        nn.Linear(int_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),

        nn.Linear(int_dim, output_dim), nn.Sigmoid()
    )
  
  def forward(self, x):
    x = x.view(-1, self.input_dim)
    x = self.net(x)
    x = x.view(-1, self.output_dims[0], self.output_dims[1], self.output_dims[2])
    return x
