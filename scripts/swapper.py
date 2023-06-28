import copy
import os
from dataclasses import dataclass
from typing import List, Union, Dict, Set, Tuple

import cv2
import numpy as np
from PIL import Image

import insightface
import onnxruntime
from scripts.roop_logging import logger
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity

from scripts.imgutils import pil_to_cv2, cv2_to_pil

providers = ["CPUExecutionProvider"]


def cosine_similarity_face(face1, face2) -> float:
    vec1 = face1.embedding.reshape(1, -1)
    vec2 = face2.embedding.reshape(1, -1)
    return max(0, cosine_similarity(vec1, vec2)[0, 0])


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


def get_faces(img_data: np.ndarray, det_size=(640, 640)):
    face_analyser = copy.deepcopy(getAnalysisModel())
    face_analyser.prepare(ctx_id=0, det_size=det_size)
    face = face_analyser.get(img_data)

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_faces(img_data, det_size=det_size_half)

    try:
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def compare_faces(img1: Image.Image, img2: Image.Image) -> float:
    face1 = get_or_default(get_faces(pil_to_cv2(img1)), 0, None)
    face2 = get_or_default(get_faces(pil_to_cv2(img2)), 0, None)

    if face1 is not None and face2 is not None:
        return cosine_similarity_face(face1, face2)
    return -1


@dataclass
class ImageResult:
    image: Image.Image
    similarity: Dict[int, float]  # face, 0..1
    ref_similarity: Dict[int, float]  # face, 0..1


def get_or_default(l, index, default):
    return l[index] if index < len(l) else default

def get_faces_from_img_files(files) :
    faces = []
    if len(files) > 0 :
        for file in files :
            print("open", file.name)
            img = Image.open(file.name)
            face = get_or_default(get_faces(pil_to_cv2(img)), 0, None)
            if face is not None :
                faces.append(face)
    return faces


def blend_faces(faces) :
    embeddings = [face.embedding for face in faces]
    if len(embeddings)> 0 :
        embedding_shape = embeddings[0].shape
        for embedding in embeddings:
            if embedding.shape != embedding_shape:
                raise ValueError("embedding shape mismatch")

        blended_embedding = np.mean(embeddings, axis=0)
        blended = faces[0]
        blended.embedding = blended_embedding
        return blended
    return None


def swap_face(
    reference_face: np.ndarray,
    source_face: np.ndarray,
    target_img: Image.Image,
    model: str,
    faces_index: Set[int] = {0},
    same_gender=True,
) -> ImageResult:
    return_result = ImageResult(target_img, {}, {})
    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    gender = source_face["gender"]
    print("Source Gender ", gender)
    if source_face is not None:
        result = target_img
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
        face_swapper = getFaceSwapModel(model_path)
        target_faces = get_faces(target_img)
        print("Target faces count", len(target_faces))

        if same_gender:
            target_faces = [x for x in target_faces if x["gender"] == gender]
            print("Target Gender Matches count", len(target_faces))

        for i, swapped_face in enumerate(target_faces):
            logger.info(f"swap face {i}")
            if i in faces_index:
                result = face_swapper.get(result, swapped_face, source_face)

        result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        return_result.image = result_image

        try:
            result_faces = get_faces(
                cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            )
            if same_gender:
                result_faces = [x for x in result_faces if x["gender"] == gender]

            for i, swapped_face in enumerate(result_faces):
                logger.info(f"compare face {i}")
                if i in faces_index and i < len(target_faces):
                    return_result.similarity[i] = cosine_similarity_face(
                        source_face, swapped_face
                    )
                    return_result.ref_similarity[i] = cosine_similarity_face(
                        reference_face, swapped_face
                    )

                print("similarity", return_result.similarity)
                print("ref similarity", return_result.ref_similarity)

        except Exception as e:
            logger.error(str(e))

    return return_result
