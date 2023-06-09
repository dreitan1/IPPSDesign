import torch
from torch import nn as nn


class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.Sigmoid(),

            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
        )

  def forward(self, x):
      res = x
      x = self.linear(x)
      x += res
      x = nn.Sigmoid()(x)
      return x


class ResNet(nn.Module):
    def __init__(self, layers=[2, 2, 2], input_dim=100*2, output_dims=(1, 20, 20)):
        super(ResNet, self).__init__()

        self.input_dim = input_dim
        self.output_dims = output_dims
        int_dim = 100
        output_dim = int(torch.prod(torch.Tensor(output_dims), 0).item())

        self.bias = nn.Parameter(torch.ones(output_dims))

        self.net = nn.Sequential(
            nn.Linear(input_dim, int_dim),
            nn.BatchNorm1d(int_dim),
            nn.Sigmoid(),

            self.make_layer(int_dim, int_dim, layers[0]),
            self.make_layer(int_dim, int_dim, layers[1]),
            self.make_layer(int_dim, int_dim, layers[2]),
        )
        self.fc = nn.Linear(int_dim, output_dim)
        
    def make_layer(self, in_channels, out_channels, blocks):
        layers = []
        layers.append(ResBlock(in_channels, out_channels))
        in_channels = out_channels
        for i in range(1, blocks):
            layers.append(ResBlock(in_channels, out_channels))
        return nn.Sequential(*layers)
      
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, self.output_dims[0], self.output_dims[1], self.output_dims[2])
        return nn.Sigmoid()(x+self.bias)
