from modules.face_restoration import FaceRestoration
from modules.upscaler import UpscalerData
from dataclasses import dataclass
from typing import List, Union, Dict, Set, Tuple
from scripts import imgutils
from scripts.roop_logging import logger
from PIL import Image
import numpy as np
from scripts import swapper, imgutils
from modules import scripts, shared, processing
from modules.processing import (Processed, StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img,
                                StableDiffusionProcessingTxt2Img)
import cv2

@dataclass
class UpscaleOptions:
    face_restorer_name: str = ""
    restorer_visibility: float = 0.5
    upscaler_name: str = ""
    scale: int = 1
    upscale_visibility: float = 0.5
    
    inpainting_denoising_strengh : float = 0
    inpainting_prompt : str = ""
    inpainting_negative_prompt : str = ""
    inpainting_steps : int = 20

    @property
    def upscaler(self) -> UpscalerData:
        for upscaler in shared.sd_upscalers:
            if upscaler.name == self.upscaler_name:
                return upscaler
        return None

    @property
    def face_restorer(self) -> FaceRestoration:
        for face_restorer in shared.face_restorers:
            if face_restorer.name() == self.face_restorer_name:
                return face_restorer
        return None


def upscale_image(image: Image.Image, upscale_options: UpscaleOptions):
    result_image = image
    try :
        if upscale_options.inpainting_denoising_strengh > 0 :
            result_image = img2img_diffusion(image, 
                                            inpainting_prompt=upscale_options.inpainting_prompt, 
                                            inpainting_negative_prompt=upscale_options.inpainting_negative_prompt, 
                                            inpainting_denoising_strength=upscale_options.inpainting_denoising_strengh,
                                            inpainting_steps=upscale_options.inpainting_steps)

        if upscale_options.upscaler is not None and upscale_options.upscaler.name != "None":
            original_image = result_image.copy()
            logger.info(
                "Upscale with %s scale = %s",
                upscale_options.upscaler.name,
                upscale_options.scale,
            )
            result_image = upscale_options.upscaler.scaler.upscale(
                image, upscale_options.scale, upscale_options.upscaler.data_path
            )
            if upscale_options.scale == 1:
                result_image = Image.blend(
                    original_image, result_image, upscale_options.upscale_visibility
                )

        if upscale_options.face_restorer is not None:
            original_image = result_image.copy()
            logger.info("Restore face with %s", upscale_options.face_restorer.name())
            numpy_image = np.array(result_image)
            numpy_image = upscale_options.face_restorer.restore(numpy_image)
            restored_image = Image.fromarray(numpy_image)
            result_image = Image.blend(
                original_image, restored_image, upscale_options.restorer_visibility
            )

    except Exception as e:
        logger.error("Failed to upscale %s", e)

    return result_image

def resize_bbox(bbox):
    x_min, y_min, x_max, y_max = bbox
    x_min = int(x_min // 8) * 8 if x_min % 8 != 0 else x_min
    y_min = int(y_min // 8) * 8 if y_min % 8 != 0 else y_min
    x_max = int(x_max // 8 + 1) * 8 if x_max % 8 != 0 else x_max
    y_max = int(y_max // 8 + 1) * 8 if y_max % 8 != 0 else y_max
    return x_min, y_min, x_max, y_max

def get_ldsr() -> UpscalerData:
    for upscaler in shared.sd_upscalers:
        if upscaler.name == "LDSR":
            return upscaler
    return None
            
def resize_small_image(img: Image.Image, min_resolution=512, use_ldsr  = True):
    width, height = img.size
    if min(width, height) > min_resolution: 
        return img
    k = float(min_resolution) / float(min(width, height))
    target_width = int(round(width * k))
    target_height = int(round(height * k))
    if not use_ldsr :
        resized_img = img.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
    else :
        logger.info("Upscale face with LDSR")
        resized_img = get_ldsr().scaler.upscale(
                img, k, get_ldsr().data_path
        )
    return resized_img

def create_mask(image, box_coords):
    width, height = image.size
    mask = Image.new("L", (width, height), 255)
    x1, y1, x2, y2 = box_coords
    for x in range(width):
        for y in range(height):
            if x1 <= x <= x2 and y1 <= y <= y2:
                mask.putpixel((x, y), 255)
            else:
                mask.putpixel((x, y), 0)
    return mask

def img2img_diffusion(img : Image.Image, inpainting_prompt : str, inpainting_denoising_strength : float = 0.1, inpainting_negative_prompt : str="", inpainting_steps : int = 20) -> Image.Image :
    try :
        logger.info("send faces to image to image")
        img = img.copy()
        faces = swapper.get_faces(imgutils.pil_to_cv2(img))
        if faces:
            for face in faces:
                bbox =face.bbox.astype(int)
                mask = create_mask(img, bbox)
                prompt = inpainting_prompt.replace("[gender]", "man" if face["gender"] == 1 else "woman")
                negative_prompt = inpainting_negative_prompt.replace("[gender]", "man" if face["gender"] == 1 else "woman")

                logger.info("Denoising prompt : %s", prompt)
                logger.info("Denoising strenght : %s", inpainting_denoising_strength)
                i2i_p = StableDiffusionProcessingImg2Img([img], steps =inpainting_steps, width = img.width, inpainting_fill=1, inpaint_full_res= True, height = img.height, mask=mask, prompt = prompt,negative_prompt=negative_prompt, denoising_strength=inpainting_denoising_strength)
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

        
