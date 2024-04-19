import cv2
from PIL import Image
from PIL.ImageTransform import QuadTransform
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob

from ocrpy.dataset.base import OCRDataset
from ocrpy.dataset import imgproc

class ICDAR2013Dataset(OCRDataset):
    def __init__(self,
                 dataroot: str = "/home/zuoyun.zheng/data/ICDARChallenge2/ICDAR2013/recognition",
                 labels_dict: dict = None,
                 image_width: int = None,
                 image_height: int = None,
                 mean: list = None,
                 std: list = None,
                 mode: str = None):
        assert mode in ["train", "test"]
        if mode == "train":
            self.annotation_file_names = ["train/gt.txt"]
        elif mode == "test":
            self.annotation_file_names = ["test/Challenge2_Test_Task3_GT.txt"]
        super(ICDAR2013Dataset, self).__init__(
            dataroot, labels_dict, image_width, image_height,
            mean, std, mode
        )

    def load_image_label_from_file(self):
        image_paths, image_targets = [], []
        # Read image path and corresponding text information
        for annotation_file_name in self.annotation_file_names:
            with open(self.dataroot / annotation_file_name, "r", encoding="UTF-8") as f:
                for line in f.readlines():
                    image_path, image_target = line.strip().split(", ")
                    image_target = image_target.strip('"')
                    # Skip empty ground truth training samples
                    if self.mode == "train" and not any([char in self.labels_dict for char in image_target]):
                        continue
                    image_paths.append(self.dataroot / self.mode / image_path)
                    image_targets.append(image_target)
        return image_paths, image_targets

class ICDAR2015Dataset(OCRDataset):
    def __init__(self,
                 dataroot: str = "/home/zuoyun.zheng/data/ICDARChallenge4",
                 labels_dict: dict = None,
                 image_width: int = None,
                 image_height: int = None,
                 range_norm: bool = True,
                 mean: list = None,
                 std: list = None,
                 mode: str = None):
        assert mode in ["train", "test"]
        if mode == "train":
            self.annotation_file_name = "train/gt" 
        elif mode == "test":
            self.annotation_file_name = "test/"
            raise NotImplementedError("No bounding box available yet so do not use test set!!")
        super(ICDAR2015Dataset, self).__init__(
            dataroot, labels_dict, image_width, image_height,
            range_norm, mean, std, mode
        )

    def load_image_label_from_file(self):
        # image_paths: [(path, [coordinate x 4]), ...]
        image_paths, image_targets = [], []
        # Read image path and corresponding text information
        for annotation_file_name in (self.dataroot / self.annotation_file_name).glob("*.txt"):
            image_path = Path(str(annotation_file_name).replace("gt/gt_", "image/")).with_suffix(".jpg")
            with open(annotation_file_name, "r", encoding="UTF-8-SIG") as f:
                for line in f.readlines():
                    line = line.strip().split(",")
                    image_coords, image_target = line[:-1], line[-1]
                    if "#" in image_target: continue
                    # Reorder from clock-wise to counter-clock-wise
                    cw2ccw = [0,1,6,7,4,5,2,3]
                    image_coords = [int(image_coords[i]) for i in cw2ccw]
                    image_paths.append((image_path, image_coords))
                    image_targets.append(image_target)
        return image_paths, image_targets

    def __getitem__(self, index: int) -> [str, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path, image_coords = self.image_paths[index]

        # Read the image and convert it to grayscale
        image = Image.open(str(image_path)).convert("L")
        # Segment and scale to the size of the image that the model can accept
        image = image.transform((self.image_width, self.image_height), QuadTransform(image_coords))
        image = np.array(image)[:, :, np.newaxis]

        # TODO: use ImageNet norm
        # Normalize and convert to Tensor format
        image = imgproc.image2tensor(image, range_norm=self.range_norm, mean=self.mean, std=self.std)

        # Read images target
        target = self.image_targets[index]

        if self.mode == "train":
            # TODO: add preprocessing steps, i.e. affine transformation, gitter, ...
            target = [self.labels_dict.get(character, 0) for character in target]
            target = torch.tensor(target, dtype=torch.long) 
            target_length = torch.tensor([len(target)], dtype=torch.long)
            return image, target, target_length
        elif self.mode == "valid" or self.mode == "test":
            return image_path, image, target
        else:
            raise ValueError("Unsupported data processing model, please use `train`, `valid` or `test`.")


if __name__=="__main__":
    from torch.utils.data import DataLoader
    from base import train_collate_fn
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    labels_dict = {char: i + 1 for i, char in enumerate(chars)}
    dataset = ICDAR2013Dataset(
         labels_dict=labels_dict,
         image_width=100, image_height=32, mode="train",
         mean=0.5, std=0.5, range_norm=True
    )
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, collate_fn=train_collate_fn)
    data = next(iter(dataloader))
    print(data, data[0].shape, len(dataset.image_paths))
    dataset = ICDAR2015Dataset(
         labels_dict=labels_dict,
         image_width=100, image_height=32, mode="train",
         mean=0.5, std=0.5, range_norm=True
    )
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, collate_fn=train_collate_fn)
    images, targets, _ = next(iter(dataloader)) 
    print(images.shape, len(dataset.image_paths))
    for idx in range(images.shape[0]):
        image = ((dataset.mean + (images[idx] * dataset.std))*255)[0].numpy().astype(np.uint8) # [H, W]
        Image.fromarray(image).save(f"icdar2015_ex{idx}.jpg")
