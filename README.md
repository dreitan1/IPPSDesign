# IPPSDesign

Repository for "Inverse Design of Silicon Photonics Components: A Study from Deep Learning Perspective"

# Results

Whole-data evaluation

<img width="723" alt="results" src="https://github.com/dreitan1/IPPSDesign/assets/79112037/442b45b8-3bf4-4528-9901-0ee9c0bf0a0c">

Our results show that our ResNet and InceptionNet models both perform reasonably well for PPS generation, especially when compared with VGG. Our results when trained on 90% of the data and tested on the remaining 10% are concurrent withthe results found in our whole-data evaluation.

# Training the Forward Model

python3 -m train_forward [--path=(default=..)]

# Training the Inverse Model

python3 -m train_inverse --model=(ResNet | InceptionNet | VGG) [--model_name=(default=res_model.pth | inception_model.pth | vgg_model.pth)] [--path=(default=..)]

If model name is not specified, defaults to name recognized by default tester, based on model used.

Input structures and output characteristics expected to be in folder at $path + "Machine learning inverse design/input_patterns" and $path + "Machine learning inverse design/output_characteristics" respectively. 

This code trains on the entire dataset. To instead run the training with a 90:10 train-test split, replace train_inverse with train_inverse9010

# Testing the Models

python3 -m test_inverse --model=(ResNet | InceptionNet | VGG) [--model_name=(default=res_model.pth | inception_model.pth | vgg_model.pth)] [--path=(default=..)]

To run this program, 2 models are required in the models directory: the inverse model to test and the forward model (forward_model.pth).

# Misc.

All models save to/load from the '/models' directory, located at the specified path. All figures save to the '/figures' directory, also located at that path.

Pre-trained models and data can be downloaded from: https://drive.google.com/drive/folders/1QRARk_RdEI-WgjzwzjKGTikWUeJLO2DH?usp=sharing
