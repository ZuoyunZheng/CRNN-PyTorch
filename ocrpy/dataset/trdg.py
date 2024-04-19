import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ocrpy.dataset.base import OCRDataset
from ocrpy.dataset import imgproc

class TRDGDataset(OCRDataset):
    def __init__(self,
                 dataroot: str = "/home/zuoyun.zheng/gitlab/ocrpy/data/trdg9/",
                 labels_dict: dict = None,
                 image_width: int = None,
                 image_height: int = None,
                 mean: list = None,
                 std: list = None,
                 mode: str = None):
        assert mode in ["train", "test", "val"]

        super(TRDGDataset, self).__init__(
            dataroot, labels_dict, image_width, image_height,
            mean, std, mode
        )

    def load_image_label_from_file(self):
        image_paths, image_targets = [], []
        for partition_label_file in tqdm(self.dataroot.rglob("labels.txt")):
            partition = partition_label_file.parent
            image_files = dict(
                [(p.name.split("_")[0], p) for p in partition.glob("*.jpg")]
            )
            label_fh = open(partition_label_file, "r")
            labels = label_fh.readlines()
            
            assert len(labels) == len(image_files), \
                f"Label file {partition_label_file} has {len(label_fh.readlines())}" + \
                f" while there are {len(image_files)} in directory"

            for f in labels:
                image_name, image_target = f.strip().split(" ")
                image_name = image_name.split(".")[0]
                image_paths.append(image_files[image_name])
                image_targets.append(image_target)
        return image_paths, image_targets

if __name__=="__main__":
    from torch.utils.data import DataLoader
    from ocrpy.dataset.base import train_collate_fn
    from PIL import Image
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    labels_dict = {char: i + 1 for i, char in enumerate(chars)}
    dataset = TRDGDataset(
         labels_dict=labels_dict,
         image_width=200, image_height=64, mode="train",
         mean=0.5, std=0.5,
    )
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True, collate_fn=train_collate_fn)
    data = next(iter(dataloader))
    print(data, data[0].shape, len(dataset.image_paths))
    images, targets, _ = next(iter(dataloader)) 
    for idx in range(images.shape[0]):
        image = ((dataset.mean + (images[idx] * dataset.std))*255)[0].numpy().astype(np.uint8) # [H, W]
        Image.fromarray(image).save(f"trdg_{idx}.jpg")
    import pdb; pdb.set_trace()
