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
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from ocrpy.dataset import imgproc

__all__ = [
    "train_collate_fn", "valid_test_collate_fn",
    "OCRDataset"
]


def train_collate_fn(batch: [torch.Tensor, torch.Tensor, torch.Tensor]) -> [torch.Tensor,
                                                                            torch.Tensor,
                                                                            torch.Tensor]:
    images, targets, target_lengths = zip(*batch)
    images = [i for i in images if i != None]
    targets = [t for t in targets if t != None]
    target_lengths = [tl for tl in target_lengths if tl != None]

    images = torch.stack(images, 0)
    #max_len = len(max(targets, key=len))
    #targets = [t + [0]*(max_len-len(t)) for t in targets]
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)

    return images, targets, target_lengths


def valid_test_collate_fn(batch: [str, torch.Tensor, str]) -> [str, torch.Tensor, str]:
    image_path, images, target = zip(*batch)
    image_path = [i for i in image_path if i != None]
    images = [i for i in images if i != None]
    target = [t for t in target if t != None]
    images = torch.stack(images, 0)

    return image_path, images, target


class OCRDataset(Dataset):
    def __init__(self,
                 dataroot: str = None,
                 labels_dict: dict = None,
                 image_width: int = None,
                 image_height: int = None,
                 mean: list = None,
                 std: list = None,
                 mode: str = None):
        self.dataroot = Path(dataroot)
        self.labels_dict = labels_dict
        self.image_width = image_width
        self.image_height = image_height
        self.mean = mean
        self.std = std
        self.mode = mode
        if self.mode == "train":
            assert self.labels_dict is not None

        self.image_paths, self.image_targets = self.load_image_label_from_file()
        self.num_skipped = 0

    def load_image_label_from_file(self):
        raise NotImplementedError
        ## Initialize the definition of image path, image text information, etc.
        #images_path = []
        #images_target = []

        ## Read image path and corresponding text information
        #with open(os.path.join(self.dataroot, self.annotation_file_name), "r", encoding="UTF-8") as f:
        #    for line in f.readlines():
        #        image_path, image_target = line.strip().split(" ")
        #        images_path.append(os.path.join(self.dataroot, image_path))
        #        images_target.append(image_target)

        #return images_path, images_target

    def __getitem__(self, index: int) -> [str, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]

        # TODO: try not grayscale
        # Read the image and convert it to grayscale
        try:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print(f"Corrupted image at path {image_path}")
            print(e)
            self.num_skipped += 1
            return None, None, None
        # TODO: collect image size statistics
        # Scale to the size of the image that the model can accept
        image = cv2.resize(image, (self.image_width, self.image_height), interpolation=cv2.INTER_CUBIC)
        image = np.reshape(image, (self.image_height, self.image_width, 1))

        # TODO: use ImageNet norm
        # Normalize and convert to Tensor format
        try:
            image = imgproc.image2tensor(image, mean=self.mean, std=self.std)
        except Exception as e:
            self.num_skipped += 1
            return None, None, None

        # Read images target
        target = self.image_targets[index].lower()

        if self.mode == "train":
            # TODO: add preprocessing steps, i.e. affine transformation, gitter, ...
            target = [self.labels_dict.get(character, 0) for character in target]
            target = [t for t in target if t != 0]
            target = torch.tensor(target, dtype=torch.long) 
            target_length = torch.tensor([len(target)], dtype=torch.long)
            return image, target, target_length
        elif self.mode == "valid" or self.mode == "test":
            return image_path, image, target
        else:
            raise ValueError("Unsupported data processing model, please use `train`, `valid` or `test`.")

    def __len__(self):
        return len(self.image_paths)

if __name__=="__main__":
    dataset = OCRDataset(
         image_width=224, image_height=224, mode="test",
         mean=0.5, std=0.5, range_norm=True
    )
    data = next(iter(dataset))
    print(data)
