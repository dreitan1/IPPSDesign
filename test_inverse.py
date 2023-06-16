import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from resnet import ResNet
from inceptionnet import InceptionNet
from vgg import VGG
from forward import FWBinaryImageCNN
import os
import argparse


# Load in dataset images and characteristics
def load_files(path):
    # Loading the Structure file of the images
    files = [f[9:-4] for f in os.listdir(path + "/input_patterns") if os.path.isfile(os.path.join(path + "/input_patterns", f))]
    char_files = [f[15:-4] for f in os.listdir(path + "/output_characteristics") if os.path.isfile(os.path.join(path + "/output_characteristics", f))]
    files = list(set(files) & set(char_files))
    # Can pick a file specifically to focus on to view structure and graphs
    # files = [4076]
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
    wavelengths = torch.zeros([100])
    for counts in range(Chars.size()[0]):
        Character_ = np.loadtxt(path + '/output_characteristics/characteristics' + str(files[counts]) + '.txt', dtype=float)

        Chars1[counts, :, :] = torch.tensor(Character_)[:, :]
        Chars[counts, :, 0] = Chars1[counts, :, 1]
        Chars[counts, :, 1] = Chars1[counts, :, 2]
        if counts == 0:
          wavelengths[:] = Chars1[counts, :, 0]

    return inputs, Chars, [float("{:.2f}".format(x.item())) for x in wavelengths*(10**9)]


# Set up arg parser
parser = argparse.ArgumentParser(
                    prog='Inverse Model Trainer',
                    description='Trains an inverse model')
parser.add_argument('--model', required=True)
parser.add_argument('--model_name', required=False, default=None)
parser.add_argument('--path', required=False, default="..")
args = parser.parse_args()

# Setup test variables
path = args.path

# wavelengths: (100x2)
# inputs: (1x20x20)
inputs, chars, wavelengths = load_files(path + "/Machine learning inverse design")

model_name = args['model_name']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.model == "ResNet":
  model = ResNet().to(device)
  if model_name is None:
    model_path = "res_model.pth"
elif args.model == "InceptionNet":
  model = InceptionNet().to(device)
  if model_name is None:
    model_path = "inception_model.pth"
elif args.model == "VGG":
   model = VGG().to(device)
   if model_name is None:
    model_path = "vgg_model.pth"
forward_model = FWBinaryImageCNN().to(device)

batch_size = 1
dataset = TensorDataset(chars, inputs)
loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=False)

model_path = path + "/models/" + model_path
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

forward_model_path = path + "/models/forward_model.pth"
forward_model.load_state_dict(torch.load(forward_model_path, map_location=device))
forward_model.eval()

# Degree of testing precision
prec = 0.1

precision = 0
accuracy = 0
total = 0

# pixel value = ceil(output - divide)
# pixel value = {0, 1}
divide = 0.5

full_test = True
# Reference number pps to show and save
ref_num = 0
save = True

os.makedirs(path + "/figures", exist_ok=True)

with torch.no_grad():
  im = np.zeros((30, 30))
  ref = np.zeros((30, 30))
  for i, (chs, ins) in enumerate(loader):
    chs = chs.to(device)
    ins = ins.to(device)

    out = model(chs)
    
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

  im = np.array(im.cpu())[0][0]
  ref = np.array(ref.cpu())[0][0]
  sample = np.array(sample.cpu())[0][0]

  norm = plt.Normalize(0, 1)

  colors = [(0, 0, 0), (0.61328125, 0.21875, 0.20703125)] # first color is black, last is red
  cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=20)
  fig = plt.figure()
  ax1 = fig.add_subplot(1, 2, 1)
  ax1.title.set_text(args.model)
  ax1.set_xticklabels([])
  ax1.set_yticklabels([])
  plt.imshow(im, cmap=cm, norm=norm)
  ax2 = fig.add_subplot(1, 2, 2)
  ax2.title.set_text("Original")
  ax2.set_xticklabels([])
  ax2.set_yticklabels([])
  plt.imshow(ref, cmap=cm, norm=norm)

  if save:
    plt.savefig(path + '/figures/Image' + str(ref_num), bbox_inches='tight')
    str_im = ""
    for y in range(im.shape[0]):
       for x in range(im.shape[1]):
          str_im += str(int(im[y][x]))
          if x != im.shape[1] - 1:
             str_im += '\t'   
       str_im += '\n'
    with open(path + '/figures/' + (args.model).lower() + '_structure.txt', 'w') as file:
       file.write(str_im)
    print("Saved PPS output", str(ref_num))
  plt.show()
  plt.close()

with torch.no_grad():
  inverse_accuracy = 0
  forward_accuracy = 0
  relative_accuracy = 0
  total = 0
  for i, (chs, ins) in enumerate(loader):
    chs = chs.to(device)
    ins = ins.to(device)
    out = forward_model(torch.ceil(model(chs) - divide))
    inverse_accuracy += ((out - chs)**2).abs().mean()
    relative_accuracy += ((out - forward_model(ins))**2).abs().mean()
    forward_accuracy += ((forward_model(ins) - chs)**2).abs().mean()
    total += 1
  inverse_accuracy = inverse_accuracy / total
  relative_accuracy = relative_accuracy / total
  forward_accuracy = forward_accuracy / total

  print(args.model + " MSE: " + "{0:.6f}".format(inverse_accuracy))
  print(args.model + " Relative MSE: " + "{0:.6f}".format(relative_accuracy))
  print("Forward MSE: " + "{0:.6f}".format(forward_accuracy))

inverseT1 = []
inverseT2 = []
forwardT1 = []
forwardT2 = []
realT1 = []
realT2 = []

# Generate 10 of these
figures_to_generate = 10
figures_to_generate = min(figures_to_generate, len(wavelengths))

font_size = 16

with torch.no_grad():
  for i, (chs, ins) in enumerate(loader):
    chs = chs.to(device)
    ins = ins.to(device)

    inverse_output = forward_model(torch.ceil(model(chs) - divide))
    forward_output = forward_model(ins)

    inverseT1.append(inverse_output[0, :, 0])
    inverseT2.append(inverse_output[0, :, 1])
    forwardT1.append(forward_output[0, :, 0])
    forwardT2.append(forward_output[0, :, 1])
    realT1.append(chs[0, :, 0])
    realT2.append(chs[0, :, 1])
    if i == figures_to_generate:
      break

for i in range(len(inverseT1)):
  print("Datapoint " + str(i))
  fig1 = plt.figure()
  ax1 = fig1.add_subplot(1, 1, 1)
  ax1.title.set_text("T1")
  ax1.set_ylim([0, 0.4])
  ax1.title.set_size(16)
  plt.plot(wavelengths, inverseT1[i].cpu(), 'bo--', label=args.model)
  #plt.plot(wavelengths, forwardT1[i].cpu(), 'go--', label='Forward Model')
  plt.plot(wavelengths, realT1[i].cpu(), 'r+--', label='Original')
  plt.legend(loc="upper right", fontsize=font_size)
  plt.xlabel("Wavelength (nm)")
  plt.ylabel("Characteristics")
  plt.savefig(path + '/figures/Datapoint' + str(i) + "T1")
  plt.show()
  plt.close()
  fig2 = plt.figure()
  ax2 = fig2.add_subplot(1, 1, 1)
  ax2.title.set_text("T2")
  ax2.set_ylim([0, 0.4])
  ax2.title.set_size(16)
  plt.plot(wavelengths, inverseT2[i].cpu(), 'bo--', label=args.model)
  #plt.plot(wavelengths, forwardT2[i].cpu(), 'go--', label='Forward Model')
  plt.plot(wavelengths, realT2[i].cpu(), 'r+--', label='Original')
  plt.legend(loc="upper right", fontsize=font_size)
  plt.xlabel("Wavelength (nm)")
  plt.ylabel("Characteristics")
  plt.savefig(path + '/figures/Datapoint' + str(i) + "T2")
  plt.show()
  plt.close()

print("Saved generated plots")
