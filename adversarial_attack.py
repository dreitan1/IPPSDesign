import torch
from torch.autograd import Variable
import torch.nn as nn


def adversarial_attack(chs, forward_model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Variable(torch.zeros(chs.shape[0], 1, 20, 20))
    output = output.to(device)
    optimizer = torch.optim.Adam([output.requires_grad_()], lr=0.1)
    criterion = nn.MSELoss()

    epochs = 1000
    for _ in range(epochs):
      optimizer.zero_grad()
      encoded = forward_model(output)
      loss = criterion(encoded, chs)
      loss.backward(retain_graph=True)
      optimizer.step()
      output.data = torch.clamp(output.data, 0, 1)
      
    return output
