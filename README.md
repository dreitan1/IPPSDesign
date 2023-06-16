# IPPSDesign

Repository for "Inverse Design of Silicon Photonics Components: A Study from Deep Learning Perspective" 

# Training the Forward Model

python3 -m train_forward [--path=(default=..)]

# Training the Inverse Model

python3 -m train_inverse <--model=(ResNet | InceptionNet | VGG)> [--model_name=(default=None)] [--path=(default=..)]

If model name is not specified, defaults to name recognized by tester.

Input structures and output characteristics expected to be in folder at $path + "/input_patterns" and $path + "/output_characteristics" respectively. 

# Testing the Models

python3 -m test_inverse <--model=(ResNEt | InceptionNet | VGG)> [--model_name=(default=None)], [--path=(default=..)]

To run this program, 2 models are required in the models directory: the inverse model to test and the forward model.

# Misc.

All models save to/load from the '/models' directory, located at the specified path. All figures save to the '/figures' directory, also located at that path.

Pre-trained models can be downloaded from: https://drive.google.com/drive/folders/1QRARk_RdEI-WgjzwzjKGTikWUeJLO2DH?usp=sharing
