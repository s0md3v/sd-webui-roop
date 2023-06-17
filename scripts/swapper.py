import copy
import math
import os
import tempfile
from dataclasses import dataclass
from typing import List, Union, Dict, Set, Tuple

import cv2
import numpy as np
from PIL import Image

import insightface
import onnxruntime
from scripts.cimage import convert_to_sd

from modules.face_restoration import FaceRestoration, restore_faces
from modules.upscaler import Upscaler, UpscalerData
from scripts.roop_logging import logger

providers = onnxruntime.get_available_providers()


@dataclass
class UpscaleOptions:
    scale: int = 1
    upscaler: UpscalerData = None
    upscale_visibility: float = 0.5
    face_restorer: FaceRestoration = None
    restorer_visibility: float = 0.5


def save_image(img: Image, filename: str):
    convert_to_sd(img).save(filename)


def cosine_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    cosine_distance = 1 - (dot_product / (norm1 * norm2))
    return cosine_distance


def cosine_similarity(test_vec: np.ndarray, source_vecs: List[np.ndarray]) -> float:
    cos_dist = sum(cosine_distance(test_vec, source_vec) for source_vec in source_vecs)
    average_cos_dist = cos_dist / len(source_vecs)
    return average_cos_dist


ANALYSIS_MODEL = None


def getAnalysisModel():
    global ANALYSIS_MODEL
    if ANALYSIS_MODEL is None:
        ANALYSIS_MODEL = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=providers
        )
    return ANALYSIS_MODEL


FS_MODEL = None
CURRENT_FS_MODEL_PATH = None


def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)

    return FS_MODEL


def upscale_image(image: Image, upscale_options: UpscaleOptions):
    result_image = image
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

    return result_image


def get_face_single(img_data: np.ndarray, face_index=0, det_size=(640, 640)):
    face_analyser = copy.deepcopy(getAnalysisModel())
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    face = face_analyser.get(img_data)

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_face_single(img_data, face_index=face_index, det_size=det_size_half)

    try:
        return sorted(face, key=lambda x: x.bbox[0])[face_index]
    except IndexError:
        return None


@dataclass
class ImageResult:
    path: Union[str, None] = None
    similarity: Union[Dict[int, float], None] = None  # face, 0..1

    def image(self) -> Union[Image.Image, None]:
        if self.path:
            return Image.open(self.path)
        return None


def swap_face(
    source_img: Image.Image,
    target_img: Image.Image,
    model: Union[str, None] = None,
    faces_index: Set[int] = {0},
    upscale_options: Union[UpscaleOptions, None] = None,
) -> ImageResult:
    fn = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    if model is not None:
        source_img = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        source_face = get_face_single(source_img, face_index=0)
        if source_face is not None:
            result = target_img
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
            face_swapper = getFaceSwapModel(model_path)

            for face_num in faces_index:
                target_face = get_face_single(target_img, face_index=face_num)
                if target_face is not None:
                    result = face_swapper.get(result, target_face, source_face)
                else:
                    logger.info(f"No target face found for {face_num}")

            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            if upscale_options is not None:
                result_image = upscale_image(result_image, upscale_options)

            save_image(result_image, fn.name)
        else:
            logger.info("No source face found")
    return ImageResult(path=fn.name)
