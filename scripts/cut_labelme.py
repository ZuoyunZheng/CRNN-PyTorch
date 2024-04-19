import argparse
import json 
from pathlib import Path

import labelme
import numpy as np
from PIL import Image
from PIL.ImageTransform import QuadTransform

def cut_images(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_label_file = open(output_dir/"gt.txt", "w")

    for image_file, label_file in zip(
        sorted(input_dir.glob("*.bmp")), sorted(input_dir.glob("*.json"))
    ):
        assert image_file.stem == label_file.stem, f"{image_file} does not correspond to {label_file}"
        image = Image.open(str(image_file)).convert("L")
        label = labelme.LabelFile(filename=label_file)
        #assert (image == np.flip(labelme.utils.img_data_to_arr(label.imageData), 2)).any(), \
        #    "Image stored in label file is different from the original, labelme problem"
        for i, polygon in enumerate(label.shapes):
            cur_label = polygon["label"]
            cur_coord = polygon["points"]
            cur_coord = [cur_coord[i] for i in [0,3,2,1]]
            cur_coord = [i for x in cur_coord for i in x]
            cur_image = image.transform((args.width, args.height), QuadTransform(cur_coord))
            cur_image_name = image_file.stem + '_' + str(i) + ".jpg"
            output_label_file.write(f"{cur_image_name}: {cur_label}\n")
            cur_image.save(output_dir / cur_image_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/DotMatrixText", help="Directory where all the labeled images live")
    parser.add_argument("--output_dir", type=str, default="data/DotMatrixTextCuts", help="Directory where all the cut labeled images live")
    parser.add_argument("--height", type=int, default=32, help="Output Height")
    parser.add_argument("--width", type=int, default=100, help="Output Height")
    args = parser.parse_args()

    cut_images(args)
