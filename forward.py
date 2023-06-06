import torch
from torch import nn as nn


# Full wavelength Forward moedl
class FWBinaryImageCNN(nn.Module):
    def __init__(self, input_dim=1*20*20, output_dims=[100, 2]):
        super(FWBinaryImageCNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = int(torch.prod(torch.Tensor(output_dims), 0).item())
        self.output_dims = output_dims
        
        # Input layer
        self.input1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        
        # Hidden layer 1
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Hidden layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)

        # Hidden layer 3
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.conv_net = nn.Sequential(
            self.input1, nn.Sigmoid(), self.conv1, self.pool1, nn.Sigmoid(),
            self.conv2, self.pool2, nn.Sigmoid(), self.conv3, self.pool3, 
            nn.Sigmoid(), 
        )
        
        # Output layer
        self.fc1 = nn.Linear(128 * 3 * 3, 128 * 5)
        self.fc2 = nn.Linear(128 * 5, 64 * 7)
        self.fc3 = nn.Linear(64 * 7, self.output_dim)

        self.fc_net = nn.Sequential(
            self.fc1, nn.ReLU(), self.fc2, nn.ReLU(), self.fc3,
        )

        # Output layer 2
        self.fc4 = nn.Linear(input_dim, self.output_dim)

    def forward(self, x):
        res = x.view(-1, self.input_dim)
        x = self.conv_net(x)
        x = x.view(-1, 128 * 3 * 3)
        x = self.fc_net(x)
        x = x.view(-1, self.output_dims[0], self.output_dims[1])
        res = self.fc4(res)
        res = res.view(-1, self.output_dims[0], self.output_dims[1])
        x = torch.sigmoid(x + res)
        return x
      
