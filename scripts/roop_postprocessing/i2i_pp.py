from modules.face_restoration import FaceRestoration
from modules.upscaler import UpscalerData
from dataclasses import dataclass
from typing import List, Union, Dict, Set, Tuple
from scripts.roop_logging import logger
from PIL import Image
import numpy as np
from modules import shared
from scripts.roop_utils import imgutils
from modules import shared, processing, codeformer_model

from modules.processing import (StableDiffusionProcessingImg2Img)
from enum import Enum

from scripts.roop_swapping import swapper


def img2img_diffusion(img : Image.Image, inpainting_prompt : str, inpainting_denoising_strength : float = 0.1, inpainting_negative_prompt : str="", inpainting_steps : int = 20, inpainting_sampler : str ="Euler") -> Image.Image :
    if inpainting_denoising_strength == 0  :
        return img

    try :
        logger.info(
f"""Inpainting face
Sampler : {inpainting_sampler}
inpainting_denoising_strength : {inpainting_denoising_strength}
inpainting_steps : {inpainting_steps}
"""
)
        if not isinstance(inpainting_sampler, str) :
            inpainting_sampler = "Euler"

        logger.info("send faces to image to image")
        img = img.copy()
        faces = swapper.get_faces(imgutils.pil_to_cv2(img))
        if faces:
            for face in faces:
                bbox =face.bbox.astype(int)
                mask = imgutils.create_mask(img, bbox)
                prompt = inpainting_prompt.replace("[gender]", "man" if face["gender"] == 1 else "woman")
                negative_prompt = inpainting_negative_prompt.replace("[gender]", "man" if face["gender"] == 1 else "woman")
                logger.info("Denoising prompt : %s", prompt)
                logger.info("Denoising strenght : %s", inpainting_denoising_strength)                
                i2i_p = StableDiffusionProcessingImg2Img([img],sampler_name=inpainting_sampler, do_not_save_samples=True, steps =inpainting_steps, width = img.width, inpainting_fill=1, inpaint_full_res= True, height = img.height, mask=mask, prompt = prompt,negative_prompt=negative_prompt, denoising_strength=inpainting_denoising_strength)
                i2i_processed = processing.process_images(i2i_p)
                images = i2i_processed.images
                if len(images) > 0 :
                    img = images[0]
        return img
    except Exception as e :
        logger.error("Failed to apply img2img to face : %s", e)
        import traceback
        traceback.print_exc()
        raise e
