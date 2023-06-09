import torch
from torch import nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
from resnet import ResNet
from inceptionnet import InceptionNet
from forward import FWBinaryImageCNN


def load_files(path):
    # Loading the Structure file of the images
    files = [f[9:-4] for f in os.listdir(path + "/input_patterns") if os.path.isfile(os.path.join(path + "/input_patterns", f))]
    char_files = [f[15:-4] for f in os.listdir(path + "/output_characteristics") if os.path.isfile(os.path.join(path + "/output_characteristics", f))]
    files = list(set(files) & set(char_files))
    sizes = len(files)
    w = 20
    h = 20
    inputs = torch.zeros([sizes, 1, w, h])
    for count in range(inputs.size()[0]):
        struct_1 = np.loadtxt(path + '/input_patterns/structure' + str(files[count]) + '.txt', dtype=int)
        inputs[count, :, :, :] = torch.tensor(struct_1).view(1, 1, w, h)

    # Loading the Character file
    Chars1 = torch.zeros([sizes, 100, 3])
    Chars = torch.zeros([sizes, 100, 2])
    for counts in range(Chars.size()[0]):
        Character_ = np.loadtxt(path + '/output_characteristics/characteristics' + str(files[counts]) + '.txt', dtype=float)

        Chars1[counts, :, :] = torch.tensor(Character_)[:, :]
        Chars[counts, :, 0] = Chars1[counts, :, 1]
        Chars[counts, :, 1] = Chars1[counts, :, 2]

    return inputs, Chars


# Computes sum of negative Bernoulli log likelihood estimator
# x = label, p = output
# Binary Cross Entropy
def BernoulliLoss(outputs: torch.Tensor, labels: torch.Tensor(), ep=1e-6) -> torch.Tensor:
    BLoss = (-1 * (torch.mul(labels, torch.log(outputs + ep)) + torch.mul((1 - labels), torch.log(1 - outputs + ep)))).mean()
    return BLoss


# Bernoulli loss + loss between difference in outputs from forward model in label and input
def BernoulliForwardLoss(outputs: torch.Tensor, labels: torch.Tensor(), forward_model, ep=1e-6) -> torch.Tensor:
    BLoss = (-1 * (torch.mul(labels, torch.log(outputs + ep)) + torch.mul((1 - labels), torch.log(1 - outputs + ep)))).mean()
    forward_loss = torch.abs((forward_model(outputs) - forward_model(labels))).mean()
    return BLoss + forward_loss


# Setup variables
path = ".."

# wavelengths: (100x2)
# inputs: (1x20x20)
inputs, chars = load_files(path + "/Machine learning inverse design")

# lr of 0.01 seems to work the best, in terms of single-pixel convergence
lr = 0.001
batch_size = 64
epochs = 15000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = BernoulliForwardLoss

dataset = TensorDataset(chars, inputs)
loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

total_steps = len(loader)
model = ResNet().to(device)
model_path = path + "/models/model.pth"
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

forward_model = FWBinaryImageCNN().to(device)
forward_path = path + "/models/forward_model.pth"
forward_model.load_state_dict(torch.load(forward_path, map_location=device))
# Set forward_model to train only
for param in forward_model.parameters():
    param.requires_grad = False
forward_model.eval()

for epoch in range(epochs):
    for i, (chs, ins) in enumerate(loader):
        chs = chs.to(device)
        ins = ins.to(device)
        optimizer.zero_grad()
        #loss = criterion(model(chs), ins)
        loss = criterion(model(chs), ins, forward_model)
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
      print(f"Epoch: [{epoch}/{epochs}], Loss: {loss.item():.6f}")

os.makedirs(path + '/models', exist_ok=True)
torch.save(model.state_dict(), model_path)
print("model complete")
