import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
import random
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

# Setup variables
path = "."

# wavelengths: (100x2)
# inputs: (1x20x20)
inputs, chars = load_files("./Machine learning inverse design")

# Train FW-Forward model
batch_size = 100
epochs = 5000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.L1Loss()

dataset = TensorDataset(chars, inputs)
loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

total_steps = len(loader)

forward_model = FWBinaryImageCNN().to(device)
forward_model_path = path + "/models/fw_forward_modelnewdata.pth"
optimizer = torch.optim.Adam(forward_model.parameters(), lr=0.000001, weight_decay=1e-6)
for epoch in range(epochs):
    for i, (chs, ins) in enumerate(loader):
        chs = chs.to(device)
        ins = ins.to(device)
        optimizer.zero_grad()
        output = forward_model(ins)
        loss = criterion(output, chs)
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch: [{epoch}/{epochs}], Loss: {loss.item():.6f}")

os.makedirs(path + '/models', exist_ok=True)
torch.save(forward_model.state_dict(), forward_model_path)
print("model complete")
