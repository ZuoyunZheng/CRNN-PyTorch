# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import argparse
from distutils.util import strtobool
import importlib.util
import os
import textwrap

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.text import CharErrorRate, WordErrorRate

from ocrpy.dataset import (
    IIIT5KDataset, ICDAR2013Dataset, MJSynthDataset,
    valid_test_collate_fn
)
from ocrpy.decoder import ctc_decode
from ocrpy.model import CRNN

def load_dataloader() -> DataLoader:
    if config.dataset == "iiit5k":
        _Dataset = IIIT5KDataset
    elif config.dataset == "ic13":
        _Dataset = ICDAR2013Dataset
    elif config.dataset == "mjsynth":
        _Dataset = MJSynthDataset
    # Load datasets
    datasets = _Dataset(
        dataroot=config.dataroot,
        image_width=config.model_image_width,
        image_height=config.model_image_height,
        mean=config.mean,
        std=config.std,
        mode="test"
    )

    dataloader = DataLoader(
        dataset=datasets,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=valid_test_collate_fn,
        pin_memory=False,
        drop_last=False,
        persistent_workers=True
    )

    return dataloader


def build_model() -> nn.Module:
    # Initialize the model
    model = CRNN(config.model_num_classes)
    model = model.to(device=config.device)
    print("Build CRNN model successfully.")

    # Load the CRNN model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load CRNN model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Start the verification mode of the model.
    model.eval()

    if config.fp16:
        # Turn on half-precision inference.
        model.half()

    return model


def main(config) -> None:
    # Initialize metrics
    cer = CharErrorRate()
    wer = WordErrorRate()
    # --- Initialize correct predictions image number
    total_correct = 0

    # Initialize model
    model = build_model()

    # Load test dataLoader
    dataloader = load_dataloader()

    # Create a experiment folder results
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # --- Get the number of test image files
    total_files = len(dataloader.dataset.image_paths)

    with open(os.path.join(config.result_dir, config.result_file_name), "w") as f:
        with torch.no_grad():
            #for image_paths, images, labels in dataloader:
            # ---
            for batch_index, (image_paths, images, labels) in enumerate(dataloader):
                # Transfer in-memory data to CUDA devices to speed up training
                images = images.to(device=config.device, non_blocking=True)
                if config.fp16:
                    # Convert to FP16
                    images = images.half()

                # Inference
                output = model(images)
                output_log_probs = F.log_softmax(output, 2)
                _, prediction_chars = ctc_decode(output_log_probs, config.chars_dict)

                # Record CER
                # TODO: implement case sensitivity
                for b in range(len(images)):
                    # ---
                    if "".join(prediction_chars[b]) == labels[b].lower():
                        total_correct += 1

                    cur_cer = cer("".join(prediction_chars[b]), labels[b].lower())
                    cur_wer = wer("".join(prediction_chars[b]), labels[b].lower())
                    if cur_cer > 0.0:
                        information = (
                            f"`{image_paths[b].name}`: "
                            f"`{labels[b].lower()}` -> "
                            f"`{''.join(prediction_chars[b])}`"
                        )
                        print(information)
                        f.write(information + "\n")

        total_cer = cer.compute()
        total_wer = wer.compute()
        information = f"CER: {total_cer *100 :.2f}%"
        information += f"\tWER: {total_wer *100 :.2f}%"
        # ---
        information += f"\tAcc: {total_correct / total_files * 100:.2f}%"
        print(information)
        # Text information to be written to the file
        f.write(information + "\n")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--name", type=str, help="Experiment name")
    # Data
    parser.add_argument("--batch", type=int, default=512, help="Batch size")
    parser.add_argument("--fp16", dest="fp16", type=lambda x: bool(strtobool(x)), default=True, help="Full Precision Inference")
    parser.add_argument("--model", type=str, help="Model checkpoint for eval")


    args = parser.parse_args()
    # Load config python script
    config_spec = importlib.util.spec_from_file_location("config", args.config)
    config = importlib.util.module_from_spec(config_spec)
    config_spec.loader.exec_module(config)
    config.exp_name = args.name
    config.result_dir =  f"./results/{config.exp_name}/test"
    config.batch_size = args.batch
    config.fp16 = args.fp16
    if args.model: config.model_path = args.model
    return config

if __name__ == "__main__":
    config = get_args() 
    main(config)
