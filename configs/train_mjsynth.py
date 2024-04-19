import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 1)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# character to be recognized
chars = "0123456789abcdefghijklmnopqrstuvwxyz!\"#$%&'()*+,-./:;?@[\\]^_`{|}~"
#chars = "0123456789abcdefghijklmnopqrstuvwxyz"
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
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "CRNN_MJSynth_adam1e-3"

if mode == "train":
    # Train dataset
    train_dataroot = ["./data/MJSYNTH"]
    train_dataset = ["mjsynth"]
    batch_size = [512]
    # Test datase
    test_dataroot = "./data/ICDARChallenge2/ICDAR2013/recognition"
    num_workers = 4

    # Incremental training and migration training
    resume = ""

    # Total num epochs
    epochs = 10
    samples = 7224612

    # Adadelta optimizer parameter
    model_lr = 0.001

    # How many iterations to print the training result
    print_frequency = 1000
