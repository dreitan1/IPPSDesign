import torch
import numpy as np
import matplotlib.pyplot as plt
from resnet import ResNet
from forward import FWBinaryImageCNN


# Load in dataset images and characteristics
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


# Setup test variables
path = ".."

# wavelengths: (100x2)
# inputs: (1x20x20)
inputs, chars = load_files(path + "/Machine learning inverse design")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet().to(device)
forward_model = FWBinaryImageCNN().to(device)

batch_size = 1
dataset = TensorDataset(chars, inputs)
loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

model_path = path + "/models/model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

forward_model_path = path + "/models/forward_model.pth"
forward_model.load_state_dict(torch.load(forward_model_path, map_location=device))
forward_model.eval()

# Degree of testing precision
prec = 0.1

precision = 0
accuracy = 0

# pixel_value = ceil(output - divide)
# pixel_value = {0, 1}
divide = 0.5

full_test = True
# Reference number pps to show and save
ref_num = 1
save = False

with torch.no_grad():
  im = np.zeros((30, 30))
  ref = np.zeros((30, 30))
  for i, (chs, ins) in enumerate(loader):
    chs = chs.to(device)
    ins = ins.to(device)

    out = model(chs)
    if i == ref_num:
        im = torch.ceil(out - divide)
        ref = ins
        sample = out
        if not full_test:
          break
    # Precision measures how close the output is to converging to either 0 or 1
    precision += torch.asarray([0.0 if prec < x < (1 - prec) else 1.0 for line in out[0][0] for x in line]).mean()
    # Accuracy measures how close the rounded pixel output is to the expected output
    accuracy += torch.abs(torch.ceil(out - divide) - ins).mean()
  precision = precision / len(loader) * 100.0
  accuracy = 100.0 - (accuracy / len(loader) * 100.0)

  print("   Accuracy = " + "{0:.2f}".format(accuracy) + "%")
  print("   Precision = " + "{0:.2f}".format(precision) + "%")

  im = np.array(im.cpu())[0][0]
  ref = np.array(ref.cpu())[0][0]
  sample = np.array(sample.cpu())[0][0]

  norm = plt.Normalize(0, 1)

  fig = plt.figure()
  ax1 = fig.add_subplot(1, 2, 1)
  ax1.title.set_text('Predicted PPS')
  plt.imshow(im, cmap="Greys_r", norm=norm)
  ax2 = fig.add_subplot(1, 2, 2)
  ax2.title.set_text('Real PPS')
  plt.imshow(ref, cmap="Greys_r", norm=norm)

  if save:
    plt.savefig(path + '/figures/Image' + str(ref_num), bbox_inches='tight')
  plt.show()
  plt.close()

print("Saved PPS output")

with torch.no_grad():
  inverse_accuracy = 0
  forward_accuracy = 0
  for i, (chs, ins) in enumerate(loader):
    chs = chs.to(device)
    ins = ins.to(device)
    inverse_accuracy += (forward_model(torch.ceil(model(chs) - divide)) - chs).abs().mean()
    forward_accuracy += (forward_model(ins) - chs).abs().mean()
  inverse_accuracy = inverse_accuracy / len(loader)
  forward_accuracy = forward_accuracy / len(loader)

  print("Inverse MAE: " + "{0:.6f}".format(inverse_accuracy))
  print("Forward MAE: " + "{0:.6f}".format(forward_accuracy))

inverseT1 = []
inverseT2 = []
forwardT1 = []
forwardT2 = []
realT1 = []
realT2 = []

# Generate 10 of these
figures_to_generate = 10

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

scale = 10

for i in range(len(inverseT1)):
  print("Datapoint " + str(i))
  print("T1")
  plt.plot(range(0, scale*len(inverseT1[i]), scale), inverseT1[i].cpu(), 'bo--', label='Inverse Model')
  plt.plot(range(0, scale*len(forwardT1[i]), scale), forwardT1[i].cpu(), 'go--', label='Forward Model')
  plt.plot(range(0, scale*len(realT1[i]), scale), realT1[i].cpu(), 'r+--', label='Real Data')
  plt.legend(loc="upper right")
  plt.savefig(path + '/figures/Datapoint' + str(i) + 'T1')
  plt.show()
  plt.close()
  print("T2")
  plt.plot(range(0, scale*len(inverseT2[i]), scale), inverseT2[i].cpu(), 'bo--', label='Inverse Model')
  plt.plot(range(0, scale*len(forwardT2[i]), scale), forwardT2[i].cpu(), 'go--', label='Forward Model')
  plt.plot(range(0, scale*len(realT2[i]), scale), realT2[i].cpu(), 'r+--', label='Real Data')
  plt.legend(loc="upper right")
  plt.savefig(path + '/figures/Datapoint' + str(i) + 'T2')
  plt.show()
  plt.close()

print("Saved generated plots")
