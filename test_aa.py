import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from forward import FWBinaryImageCNN
import os
from adversarial_attack import adversarial_attack


# Load in dataset images and characteristics
def load_files(path):
    # Loading the Structure file of the images
    files = [f[9:-4] for f in os.listdir(path + "/input_patterns") if os.path.isfile(os.path.join(path + "/input_patterns", f))]
    char_files = [f[15:-4] for f in os.listdir(path + "/output_characteristics") if os.path.isfile(os.path.join(path + "/output_characteristics", f))]
    files = list(set(files) & set(char_files))
    files = [3]
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


# Setup test variables
path = ".."

# wavelengths: (100x2)
# inputs: (1x20x20)
inputs, chars = load_files(path + "/Machine learning inverse design")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
forward_model = FWBinaryImageCNN().to(device)
forward_model_path = path + "/models/forward_model.pth"
forward_model.load_state_dict(torch.load(forward_model_path, map_location=device))

batch_size = 1
dataset = TensorDataset(chars, inputs)
loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False)

# Degree of testing precision
prec = 0.1

precision = 0
accuracy = 0
total = 0

# pixel_value = ceil(output - divide)
# pixel_value = {0, 1}
divide = 0.5

full_test = False
# Reference number pps to show and save
ref_num = 0
save = True

os.makedirs(path + "/figures", exist_ok=True)

im = np.zeros((30, 30))
ref = np.zeros((30, 30))
for i, (chs, ins) in enumerate(loader):
   chs = chs.to(device)
   ins = ins.to(device)
   
   out = adversarial_attack(chs, forward_model)

   # Precision measures how close the output is to converging to either 0 or 1
   precision += torch.asarray([0.0 if prec < x < (1 - prec) else 1.0 for line in out[0][0] for x in line]).mean()
   # Accuracy measures how close the rounded pixel output is to the expected output
   accuracy += torch.abs(torch.ceil(out - divide) - ins).mean()
   total += 1
   
   if i == ref_num:
      im = torch.ceil(out - divide)
      ref = ins
      sample = out
      if not full_test:
         break

precision = precision / total * 100.0
accuracy = 100.0 - (accuracy / total * 100.0)

print("   Accuracy = " + "{0:.2f}".format(accuracy) + "%")
print("   Precision = " + "{0:.2f}".format(precision) + "%")
  
im = np.array(im.detach().cpu())[0][0]
ref = np.array(ref.detach().cpu())[0][0]
sample = np.array(sample.detach().cpu())[0][0]

norm = plt.Normalize(0, 1)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.title.set_text('Adversarial Attack')
ax1.set_xticklabels([])
ax1.set_yticklabels([])
plt.imshow(im, cmap="Greys_r", norm=norm)
ax2 = fig.add_subplot(1, 2, 2)
ax2.title.set_text('Original')
ax2.set_xticklabels([])
ax2.set_yticklabels([])
plt.imshow(ref, cmap="Greys_r", norm=norm)

if save:
   plt.savefig(path + '/figures/Image' + str(ref_num), bbox_inches='tight')
   print("Saved PPS output", str(ref_num))
plt.show()
plt.close()

# inverse_accuracy = 0
# forward_accuracy = 0
# for i, (chs, ins) in enumerate(loader):
#    chs = chs.to(device)
#    ins = ins.to(device)
#    inverse_accuracy += (forward_model(torch.ceil(adversarial_attack(chs, forward_model) - divide)) - chs).abs().mean()
#    forward_accuracy += (forward_model(ins) - chs).abs().mean()
# inverse_accuracy = inverse_accuracy / len(loader)
# forward_accuracy = forward_accuracy / len(loader)

# print("Inverse MAE: " + "{0:.6f}".format(inverse_accuracy))
# print("Forward MAE: " + "{0:.6f}".format(forward_accuracy))

inverseT1 = []
inverseT2 = []
forwardT1 = []
forwardT2 = []
realT1 = []
realT2 = []

# Generate n of these
figures_to_generate = 1


for i, (chs, ins) in enumerate(loader):
   chs = chs.to(device)
   ins = ins.to(device)
   
   inverse_output = forward_model(torch.ceil(adversarial_attack(chs, forward_model) - divide))
   forward_output = forward_model(ins)
   
   inverseT1.append(inverse_output.detach().cpu()[0, :, 0])
   inverseT2.append(inverse_output.detach().cpu()[0, :, 1])
   forwardT1.append(forward_output.detach().cpu()[0, :, 0])
   forwardT2.append(forward_output.detach().cpu()[0, :, 1])
   realT1.append(chs.detach().cpu()[0, :, 0])
   realT2.append(chs.detach().cpu()[0, :, 1])
   if i == figures_to_generate:
      break

scale = 10

for i in range(len(inverseT1)):
  print("Datapoint " + str(i))
  fig1 = plt.figure()
  ax1 = fig1.add_subplot(1, 1, 1)
  ax1.title.set_text("T1")
  ax1.set_xticklabels([])
  plt.plot(range(0, scale*len(inverseT1[i]), scale), inverseT1[i].cpu(), 'bo--', label='Adversarial Attack')
  plt.plot(range(0, scale*len(forwardT1[i]), scale), forwardT1[i].cpu(), 'go--', label='Forward Model')
  plt.plot(range(0, scale*len(realT1[i]), scale), realT1[i].cpu(), 'r+--', label='Original')
  plt.legend(loc="upper right")
  plt.savefig(path + '/figures/Datapoint' + str(i) + "T1")
  plt.show()
  plt.close()
  fig2 = plt.figure()
  ax2 = fig2.add_subplot(1, 1, 1)
  ax2.title.set_text("T2")
  ax2.set_xticklabels([])
  plt.plot(range(0, scale*len(inverseT2[i]), scale), inverseT2[i].cpu(), 'bo--', label='Adversarial Attack')
  plt.plot(range(0, scale*len(forwardT2[i]), scale), forwardT2[i].cpu(), 'go--', label='Forward Model')
  plt.plot(range(0, scale*len(realT2[i]), scale), realT2[i].cpu(), 'r+--', label='Original')
  plt.legend(loc="upper right")
  plt.savefig(path + '/figures/Datapoint' + str(i) + "T2")
  plt.show()
  plt.close()

print("Saved generated plots")
