from typing import Tuple, List

import numpy as np
from PIL import Image
from pathlib import Path


def mean_and_std(images_list: List[Path], dimension: int):
    size = len(images_list)
    meanTotal = 0
    stdTotal = 0
    for image_path in images_list:
        img = Image.open(image_path)
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            img_np = img_np[:, :, dimension]
        mean, std = np.mean(img_np), np.std(img_np)
        meanTotal = meanTotal + mean
        stdTotal = stdTotal + std
    return meanTotal / size, stdTotal / size


def mean_and_std_from_data_path(
    data_path: str,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    images_path = Path(data_path) / "imgs"
    all_images = list(images_path.iterdir())
    mean0, std0 = mean_and_std(all_images, 0)
    mean1, std1 = mean_and_std(all_images, 1)
    mean2, std2 = mean_and_std(all_images, 2)
    return (mean0, mean1, mean2), (std0, std1, std2)


if __name__ == "__main__":
    data_path = Path("../../temp/0lmu+heidelberg")
    print(mean_and_std_from_data_path(data_path))
