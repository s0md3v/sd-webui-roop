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
from pprint import pformat
from modules.processing import (StableDiffusionProcessingImg2Img)
from enum import Enum
from scripts.roop_postprocessing.postprocessing_options import PostProcessingOptions, InpaintingWhen
from modules import sd_models

from scripts.roop_swapping import swapper


def img2img_diffusion(img : Image.Image, pp: PostProcessingOptions) -> Image.Image :
    if pp.inpainting_denoising_strengh == 0  :
        return img

    try :
        logger.info(
f"""Inpainting face
Sampler : {pp.inpainting_sampler}
inpainting_denoising_strength : {pp.inpainting_denoising_strengh}
inpainting_steps : {pp.inpainting_steps}
"""
)
        if not isinstance(pp.inpainting_sampler, str) :
            inpainting_sampler = "Euler"

        logger.info("send faces to image to image")
        img = img.copy()
        faces = swapper.get_faces(imgutils.pil_to_cv2(img))
        if faces:
            for face in faces:
                bbox =face.bbox.astype(int)
                mask = imgutils.create_mask(img, bbox)
                prompt = pp.inpainting_prompt.replace("[gender]", "man" if face["gender"] == 1 else "woman")
                negative_prompt = pp.inpainting_negative_prompt.replace("[gender]", "man" if face["gender"] == 1 else "woman")
                logger.info("Denoising prompt : %s", prompt)
                logger.info("Denoising strenght : %s", pp.inpainting_denoising_strengh)                
                
                i2i_kwargs = {"sampler_name" :pp.inpainting_sampler,
                        "do_not_save_samples":True, 
                        "steps" :pp.inpainting_steps,
                        "width" : img.width,
                        "inpainting_fill":1,
                        "inpaint_full_res":True,
                        "height" : img.height,
                        "mask": mask,
                        "prompt" : prompt,
                        "negative_prompt" :negative_prompt,
                        "denoising_strength" :pp.inpainting_denoising_strengh}
                current_model_checkpoint = shared.opts.sd_model_checkpoint
                if pp.inpainting_model and pp.inpainting_model != "Current" :
                    # Change checkpoint
                    shared.opts.sd_model_checkpoint = pp.inpainting_model
                    sd_models.select_checkpoint
                    sd_models.load_model()
                i2i_p = StableDiffusionProcessingImg2Img([img], **i2i_kwargs)
                i2i_processed = processing.process_images(i2i_p)
                if pp.inpainting_model and pp.inpainting_model != "Current" :
                    # Restore checkpoint
                    shared.opts.sd_model_checkpoint = current_model_checkpoint
                    sd_models.select_checkpoint
                    sd_models.load_model()

                images = i2i_processed.images
                if len(images) > 0 :
                    img = images[0]
        return img
    except Exception as e :
        logger.error("Failed to apply img2img to face : %s", e)
        import traceback
        traceback.print_exc()
        raise e
