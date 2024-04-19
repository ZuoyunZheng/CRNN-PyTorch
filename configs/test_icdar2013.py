import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cpu", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# character to be recognized
chars = "0123456789abcdefghijklmnopqrstuvwxyz"
labels_dict = {char: i + 1 for i, char in enumerate(chars)}
chars_dict = {label: char for char, label in labels_dict.items()}
# Model parameter configuration
model_num_classes = len(chars) + 1
model_image_width = 100
model_image_height = 32
# Mean and std of the model input data source
mean = 0.5
std = 0.5
# Current configuration parameter method
mode = "test"
# Experiment name, easy to save weights and log files
exp_name = "CRNN_MJSynth_pretrained"

batch_size = 1
# Whether to enable half-precision inference
fp16 = False

# The path and name of the folder where the verification results are saved
result_dir = f"./results/{exp_name}/test"
result_file_name = "IC13_result.txt"

# The directory path where the dataset to be verified is located
dataroot = "./data/ICDARChallenge2/ICDAR2013/recognition"
dataset = "ic13"

model_path = "results/pretrained_models/CRNN-MJSynth-e9341ede.pth.tar"
