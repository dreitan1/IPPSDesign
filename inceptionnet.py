import torch
import torch.nn as nn
from math import ceil


class InceptionBlock(nn.Module):
  def __init__(self, in_channels):
    super(InceptionBlock, self).__init__()
    self.net1 = nn.Linear(in_channels, ceil(in_channels/2))
    self.net2 = nn.Linear(in_channels, int(in_channels/4))
    self.net3 = nn.Linear(in_channels, ceil(in_channels/12))
    self.net4 = nn.Linear(in_channels, int(in_channels/6))

  def forward(self, x):
    x1 = self.net1(x)
    x2 = self.net2(x)
    x3 = self.net3(x)
    x4 = self.net4(x)
    return torch.cat([x1, x2, x3, x4], dim=1)


class InceptionNet(nn.Module):
  def __init__(self, input_dim=100*2, output_dims=[1, 20, 20]):
    super(InceptionNet, self).__init__()
    self.input_dim = input_dim
    self.output_dim = int(torch.prod(torch.Tensor(output_dims), 0).item())
    self.output_dims = output_dims

    int_dim = 100

    self.net = nn.Sequential(
        nn.Linear(input_dim, int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),

        InceptionBlock(int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        InceptionBlock(int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        InceptionBlock(int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        InceptionBlock(int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        InceptionBlock(int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        InceptionBlock(int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        InceptionBlock(int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        InceptionBlock(int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),
        InceptionBlock(int_dim), nn.BatchNorm1d(int_dim), nn.Sigmoid(),

        nn.Linear(int_dim, self.output_dim), nn.Sigmoid()
    )
  
  def forward(self, x):
    x = x.view(-1, self.input_dim)
    x = self.net(x)
    x = x.view(-1, self.output_dims[0], self.output_dims[1], self.output_dims[2])
    return x