import sys
import cv2
from pathlib import Path

import pdb; pdb.set_trace()
image_dir = Path(sys.argv[1])
for image_path in image_dir.glob("*.jpg"):
    image = cv2.imread(str(image_path))
    height, width = image.shape[:-1]
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(str(image_path), image)
    # also rotate bounding box annotations
