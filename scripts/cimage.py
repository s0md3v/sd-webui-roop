from typing import List, Union, Dict, Set, Tuple

import torch
from PIL import Image, ImageFilter
import numpy as np

def numpy_to_pil(images: np.ndarray) -> List[Image.Image]:
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def check_image(x_image: np.ndarray) -> Tuple[np.ndarray, List[bool]]:
    return x_image, [False] * len(x_image)


def check_batch(x: torch.Tensor) -> torch.Tensor:
    return x


def convert_to_sd(img: Image) -> Image:
    return img
