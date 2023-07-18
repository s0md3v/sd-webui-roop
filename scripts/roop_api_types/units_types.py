from scripts.roop_swapping import swapper
import numpy as np
import base64
import io
from dataclasses import dataclass, fields
from typing import Dict, List, Set, Tuple, Union, Optional
import dill as pickle
import gradio as gr
from insightface.app.common import Face
from PIL import Image
from scripts.roop_utils.imgutils import (pil_to_cv2,convert_to_sd)
from scripts.roop_logging import logger


@dataclass
class FaceSwapUnitDTO :
    # The image given in reference
    source_img: str
    # The checkpoint file
    source_face : str
    # base64 batch source images
    batch_files: List[str]

    # Will blend faces if True
    blend_faces: bool
    # Enable this unit
    enable: bool
    # Use same gender filtering
    same_gender: bool

    # If True, discard images with low similarity
    check_similarity : bool 
    # if True will compute similarity and add it to the image info
    compute_similarity :bool

    # Minimum similarity against the used face (reference, batch or checkpoint)
    min_sim: float
    # Minimum similarity against the reference (reference or checkpoint if checkpoint is given)
    min_ref_sim: float
    # The face index to use for swapping
    faces_index: int
    
    # Swap in the source image in img2img (before processing)
    swap_in_source: bool
    # Swap in the generated image in img2img (always on for txt2img)
    swap_in_generated: bool    