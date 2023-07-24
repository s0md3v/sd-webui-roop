from modules.face_restoration import FaceRestoration
from scripts.roop_logging import logger
from PIL import Image
from scripts.roop_postprocessing.postprocessing_options import PostProcessingOptions, InpaintingWhen
from scripts.roop_postprocessing.i2i_pp import img2img_diffusion
from scripts.roop_postprocessing.upscaling import upscale_img, restore_face


def enhance_image(image: Image.Image, pp_options: PostProcessingOptions) -> Image.Image:
    result_image = image
    try :
        if pp_options.inpainting_when == InpaintingWhen.BEFORE_UPSCALING.value :
            result_image = img2img_diffusion(image, pp_options)
        result_image = upscale_img(result_image, pp_options)

        if pp_options.inpainting_when == InpaintingWhen.BEFORE_RESTORE_FACE.value :
            result_image = img2img_diffusion(image,pp_options)

        result_image = restore_face(result_image, pp_options)
              
        if pp_options.inpainting_when == InpaintingWhen.AFTER_ALL.value :
            result_image = img2img_diffusion(image,pp_options)

    except Exception as e:
        logger.error("Failed to upscale %s", e)

    return result_image