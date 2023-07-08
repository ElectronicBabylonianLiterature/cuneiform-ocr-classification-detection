import numpy as np
from PIL import ImageOps
from mmcv import BaseTransform
from mmengine import TRANSFORMS
from torchvision.transforms import Grayscale as Grayscale_
from PIL import Image

@TRANSFORMS.register_module()
class Grayscale(BaseTransform):
    def __init__(self):
        super().__init__()

    def transform(self, results):
        img =Image.fromarray(results["img"])
        img_grayscale = ImageOps.grayscale(img)
        asd = np.array(img_grayscale)
        results["img"] = np.array(img_grayscale)
        return results

