from typing import List, Union, Dict, Set, Tuple

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import AutoFeatureExtractor
import torch
from PIL import Image, ImageFilter
import numpy as np

safety_model_id: str = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor: AutoFeatureExtractor = None
safety_checker: StableDiffusionSafetyChecker = None


def numpy_to_pil(images: np.ndarray) -> List[Image.Image]:
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def check_image(x_image: np.ndarray) -> Tuple[np.ndarray, List[bool]]:
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    safety_checker_input = safety_feature_extractor(
        images=numpy_to_pil(x_image), return_tensors="pt"
    )
    x_checked_image, hs = safety_checker(
        images=x_image, clip_input=safety_checker_input.pixel_values
    )

    return x_checked_image, hs


def check_batch(x: torch.Tensor) -> torch.Tensor:
    x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    x_checked_image, _ = check_image(x_samples_ddim_numpy)
    x = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
    return x


def convert_to_sd(img: Image) -> Image:
    _, hs = check_image(np.array(img))
    if any(hs):
        img = (
            img.resize((int(img.width * 0.1), int(img.height * 0.1)))
            .resize(img.size, Image.BOX)
            .filter(ImageFilter.BLUR)
        )
    return img
