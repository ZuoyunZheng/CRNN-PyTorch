# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
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

batch_size = 32
# Whether to enable half-precision inference
fp16 = True

# The path and name of the folder where the verification results are saved
result_dir = f"./results/{exp_name}/test"
result_file_name = "IIIT5K_result.txt"

# The directory path where the dataset to be verified is located
dataroot = "./data/IIIT5K"
dataset = "iiit5k"

model_path = "results/pretrained_models/CRNN-MJSynth-e9341ede.pth.tar"
