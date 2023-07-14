import copy
import os
from dataclasses import dataclass
from pprint import pprint
from typing import Dict, List, Set, Tuple, Union, Optional

import cv2
import insightface
import numpy as np
from insightface.app.common import Face

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

from scripts.roop_swapping import upscaled_inswapper
from scripts.roop_utils.imgutils import cv2_to_pil, pil_to_cv2
from scripts.roop_logging import logger
from scripts import roop_globals
from modules.shared import opts

providers = ["CPUExecutionProvider"]


def cosine_similarity_face(face1, face2) -> float:
    """
    Calculates the cosine similarity between two face embeddings.

    Args:
        face1 (Face): The first face object containing an embedding.
        face2 (Face): The second face object containing an embedding.

    Returns:
        float: The cosine similarity between the face embeddings.

    Note:
        The cosine similarity ranges from 0 to 1, where 1 indicates identical embeddings and 0 indicates completely
        dissimilar embeddings. In this implementation, the similarity is clamped to a minimum value of 0 to ensure a
        non-negative similarity score.
    """
    # Reshape the face embeddings to have a shape of (1, -1)
    vec1 = face1.embedding.reshape(1, -1)
    vec2 = face2.embedding.reshape(1, -1)

    # Calculate the cosine similarity between the reshaped embeddings
    similarity = cosine_similarity(vec1, vec2)

    # Return the maximum of 0 and the calculated similarity as the final similarity score
    return max(0, similarity[0, 0])

def compare_faces(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compares the similarity between two faces extracted from images using cosine similarity.
    
    Args:
        img1: The first image containing a face.
        img2: The second image containing a face.
    
    Returns:
        A float value representing the similarity between the two faces (0 to 1). 
        Returns -1 if one or both of the images do not contain any faces.
    """
    
    # Extract faces from the images
    face1 = get_or_default(get_faces(pil_to_cv2(img1)), 0, None)
    face2 = get_or_default(get_faces(pil_to_cv2(img2)), 0, None)

    # Check if both faces are detected
    if face1 is not None and face2 is not None:
        # Calculate the cosine similarity between the faces
        return cosine_similarity_face(face1, face2)
    
    # Return -1 if one or both of the images do not contain any faces
    return -1



# Global variable to store the analysis model
ANALYSIS_MODEL = None


class FaceModelException(Exception):
    pass

def getAnalysisModel():
    """
    Retrieves the analysis model for face analysis.
    
    Returns:
        insightface.app.FaceAnalysis: The analysis model for face analysis.
    """
    global ANALYSIS_MODEL
    
    # Check if the analysis model has been initialized
    if ANALYSIS_MODEL is None:
        try :
            if not os.path.exists(roop_globals.ANALYZER_DIR):
                os.makedirs(roop_globals.ANALYZER_DIR)

            logger.info("Load analysis model, will take some time.")
            # Initialize the analysis model with the specified name and providers
            ANALYSIS_MODEL = insightface.app.FaceAnalysis(
                name="buffalo_l", providers=providers, root=roop_globals.ANALYZER_DIR
            )
        except Exception as e :
            logger.error("Loading of swapping model failed, please check the requirements (On Windows, download and install Visual Studio. During the install, make sure to include the Python and C++ packages.)")
            raise FaceModelException()
    # Return the analysis model
    return ANALYSIS_MODEL


FS_MODEL = None  # Global variable to store the face swap model.
CURRENT_FS_MODEL_PATH = None  # Global variable to store the current path of the face swap model.

def getFaceSwapModel(model_path: str):
    """
    Retrieves the face swap model and initializes it if necessary.

    Args:
        model_path (str): Path to the face swap model.

    Returns:
        insightface.model_zoo.FaceModel: The face swap model.
    """
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH

    # Check if the current model path is different from the new model path.
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        logger.info("Load swapping model, will take some time.")
        try :
            CURRENT_FS_MODEL_PATH = model_path
            # Initializes the face swap model using the specified model path.
            FS_MODEL = upscaled_inswapper.UpscaledINSwapper(insightface.model_zoo.get_model(model_path, providers=providers))
        except Exception as e :
            logger.error("Loading of swapping model failed, please check the requirements (On Windows, download and install Visual Studio. During the install, make sure to include the Python and C++ packages.)")
    return FS_MODEL


def get_faces(img_data: np.ndarray, det_size=(640, 640), det_thresh : Optional[int]=None) -> List[Face]:
    """
    Detects and retrieves faces from an image using an analysis model.

    Args:
        img_data (np.ndarray): The image data as a NumPy array.
        det_size (tuple): The desired detection size (width, height). Defaults to (640, 640).

    Returns:
        list: A list of detected faces, sorted by their x-coordinate of the bounding box.
    """

    if det_thresh is None : 
        det_thresh = opts.data.get("roop_detection_threshold", 0.5)

    # Create a deep copy of the analysis model (otherwise det_size is attached to the analysis model and can't be changed)
    face_analyser = copy.deepcopy(getAnalysisModel())

    # Prepare the analysis model for face detection with the specified detection size
    face_analyser.prepare(ctx_id=0, det_thresh=det_thresh, det_size=det_size)

    # Get the detected faces from the image using the analysis model
    face = face_analyser.get(img_data)

    # If no faces are detected and the detection size is larger than 320x320,
    # recursively call the function with a smaller detection size
    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_faces(img_data, det_size=det_size_half, det_thresh=det_thresh)

    try:
        # Sort the detected faces based on their x-coordinate of the bounding box
        return sorted(face, key=lambda x: x.bbox[0])
    except Exception as e:
        return []



@dataclass
class ImageResult:
    """
    Represents the result of an image swap operation
    """

    image: Image.Image
    """
    The image object with the swapped face
    """

    similarity: Dict[int, float]
    """
    A dictionary mapping face indices to their similarity scores.
    The similarity scores are represented as floating-point values between 0 and 1.
    """

    ref_similarity: Dict[int, float]
    """
    A dictionary mapping face indices to their similarity scores compared to a reference image.
    The similarity scores are represented as floating-point values between 0 and 1.
    """


def get_or_default(l, index, default):
    """
    Retrieve the value at the specified index from the given list.
    If the index is out of bounds, return the default value instead.

    Args:
        l (list): The input list.
        index (int): The index to retrieve the value from.
        default: The default value to return if the index is out of bounds.

    Returns:
        The value at the specified index if it exists, otherwise the default value.
    """
    return l[index] if index < len(l) else default


def get_faces_from_img_files(files):
    """
    Extracts faces from a list of image files.

    Args:
        files (list): A list of file objects representing image files.

    Returns:
        list: A list of detected faces.

    """

    faces = []

    if len(files) > 0:
        for file in files:
            img = Image.open(file.name)  # Open the image file
            face = get_or_default(get_faces(pil_to_cv2(img)), 0, None)  # Extract faces from the image
            if face is not None:
                faces.append(face)  # Add the detected face to the list of faces

    return faces

def blend_faces(faces: List[Face]) -> Face:
    """
    Blends the embeddings of multiple faces into a single face.

    Args:
        faces (List[Face]): List of Face objects.

    Returns:
        Face: The blended Face object with the averaged embedding.
              Returns None if the input list is empty.
              
    Raises:
        ValueError: If the embeddings have different shapes.

    """
    embeddings = [face.embedding for face in faces]
    
    if len(embeddings) > 0:
        embedding_shape = embeddings[0].shape
        
        # Check if all embeddings have the same shape
        for embedding in embeddings:
            if embedding.shape != embedding_shape:
                raise ValueError("embedding shape mismatch")

        # Compute the mean of all embeddings
        blended_embedding = np.mean(embeddings, axis=0)
        
        # Create a new Face object using the properties of the first face in the list
        # Assign the blended embedding to the blended Face object
        blended = Face(embedding=blended_embedding, gender=faces[0].gender, age=faces[0].age)

        assert not np.array_equal(blended.embedding,faces[0].embedding) if len(faces) > 1 else True, "If len(faces)>0, the blended embedding should not be the same than the first image"
        
        return blended
    
    # Return None if the input list is empty
    return None


def swap_face(
    reference_face: np.ndarray,
    source_face: np.ndarray,
    target_img: Image.Image,
    model: str,
    faces_index: Set[int] = {0},
    same_gender=True,
    upscaled_swapper = False,
    compute_similarity = True
) -> ImageResult:
    """
    Swaps faces in the target image with the source face.

    Args:
        reference_face (np.ndarray): The reference face used for similarity comparison.
        source_face (np.ndarray): The source face to be swapped.
        target_img (Image.Image): The target image to swap faces in.
        model (str): Path to the face swap model.
        faces_index (Set[int], optional): Set of indices specifying which faces to swap. Defaults to {0}.
        same_gender (bool, optional): If True, only swap faces with the same gender as the source face. Defaults to True.

    Returns:
        ImageResult: An object containing the swapped image and similarity scores.

    """    
    return_result = ImageResult(target_img, {}, {})
    try :
        target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        gender = source_face["gender"]
        logger.info("Source Gender %s", gender)
        if source_face is not None:
            result = target_img
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
            face_swapper = getFaceSwapModel(model_path)
            target_faces = get_faces(target_img)
            logger.info("Target faces count : %s", len(target_faces))

            if same_gender:
                target_faces = [x for x in target_faces if x["gender"] == gender]
                logger.info("Target Gender Matches count %s", len(target_faces))

            for i, swapped_face in enumerate(target_faces):
                logger.info(f"swap face {i}")
                if i in faces_index:
                    result = face_swapper.get(result, swapped_face, source_face, upscale = upscaled_swapper)

            result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            return_result.image = result_image


            if compute_similarity :
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

                        logger.info(f"similarity {return_result.similarity}")
                        logger.info(f"ref similarity {return_result.ref_similarity}")

                except Exception as e:
                    logger.error("Similarity processing failed %s", e)
                    raise e
    except Exception as e :
        logger.error("Conversion failed %s", e)
        raise e
    return return_result