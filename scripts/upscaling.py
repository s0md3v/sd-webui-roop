from modules.face_restoration import FaceRestoration
from modules.upscaler import UpscalerData
from dataclasses import dataclass
from typing import List, Union, Dict, Set, Tuple
from scripts.roop_logging import logger
from PIL import Image
import numpy as np
from modules import scripts, shared


@dataclass
class UpscaleOptions:
    face_restorer_name: str = ""
    restorer_visibility: float = 0.5
    upscaler_name: str = ""
    scale: int = 1
    upscale_visibility: float = 0.5

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
        logger.info("Failed to upscale %s", e)

    return result_image
