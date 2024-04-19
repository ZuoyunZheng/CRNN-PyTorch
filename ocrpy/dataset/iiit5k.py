import cv2
import numpy as np
from scipy.io import loadmat
from pathlib import Path

from ocrpy.dataset.base import OCRDataset
from ocrpy.dataset import imgproc

class IIIT5KDataset(OCRDataset):
    def __init__(self,
                 dataroot: str = "/home/zuoyun.zheng/data/iiit5k/IIIT5K",
                 labels_dict: dict = None,
                 image_width: int = None,
                 image_height: int = None,
                 mean: list = None,
                 std: list = None,
                 mode: str = None):
        assert mode in ["train", "test"]
        self.annotation_file_names = ["trainCharBound.mat"]
        if mode == "test":
            self.annotation_file_names += ["testCharBound.mat"]

        super(IIIT5KDataset, self).__init__(
            dataroot, labels_dict, image_width, image_height,
            mean, std, mode
        )

    def load_image_label_from_file(self):
        image_paths, image_targets = [], []
        for annotation_file_name in self.annotation_file_names:
            # Read image path and corresponding text information
            mat_name = annotation_file_name[:-4]
            mat = loadmat(self.dataroot / annotation_file_name)[mat_name][0]
            image_paths.extend([self.dataroot / n[0] for n in mat["ImgName"]])
            image_targets.extend([c[0] for c in mat["chars"]])
        return image_paths, image_targets

if __name__=="__main__":
    dataset = IIIT5KDataset(
         image_width=224, image_height=224, mode="test",
         mean=0.5, std=0.5,
    )
    data = next(iter(dataset))
    print(data)
    import pdb; pdb.set_trace()
