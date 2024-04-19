import cv2
import numpy as np
from pathlib import Path

from ocrpy.dataset.base import OCRDataset
from ocrpy.dataset import imgproc

class MJSynthDataset(OCRDataset):
    def __init__(self,
                 dataroot: str = "/home/zuoyun.zheng/data/mjsynth/90kDICT32px",
                 labels_dict: dict = None,
                 image_width: int = None,
                 image_height: int = None,
                 mean: list = None,
                 std: list = None,
                 mode: str = None):
        assert mode in ["train", "test", "val"]
        self.annotation_file_name = f"annotation_{mode}.txt"

        super(MJSynthDataset, self).__init__(
            dataroot, labels_dict, image_width, image_height,
            mean, std, mode
        )

    def load_image_label_from_file(self):
        image_paths, image_targets = [], []
        # Read image path and corresponding text information
        with open(self.dataroot / self.annotation_file_name, "r", encoding="UTF-8") as f:
            for line in f.readlines():
                image_path = line.strip().split(" ")[0]
                image_target = image_path.split("_")[1]
                # Skip empty ground truth training samples
                #if self.mode == "train" and not any([char in self.labels_dict for char in image_target]):
                #    continue
                image_paths.append(self.dataroot / image_path)
                image_targets.append(image_target)

        return image_paths, image_targets

if __name__=="__main__":
    from torch.utils.data import DataLoader
    from ocrpy.dataset.base import train_collate_fn
    from PIL import Image
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    labels_dict = {char: i + 1 for i, char in enumerate(chars)}
    dataset = MJSynthDataset(
         labels_dict=labels_dict,
         image_width=100, image_height=32, mode="train",
         mean=0.5, std=0.5,
    )
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, collate_fn=train_collate_fn)
    data = next(iter(dataloader))
    print(data, data[0].shape, len(dataset.image_paths))
    images, targets, _ = next(iter(dataloader)) 
    for idx in range(images.shape[0]):
        image = ((dataset.mean + (images[idx] * dataset.std))*255)[0].numpy().astype(np.uint8) # [H, W]
        Image.fromarray(image).save(f"mjsynth_ex{idx}.jpg")
    import pdb; pdb.set_trace()
