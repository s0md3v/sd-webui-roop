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
from scripts.roop_utils.imgutils import (pil_to_cv2,convert_to_sd, base64_to_pil)
from scripts.roop_logging import logger
from pydantic import BaseModel, Field
from scripts.roop_postprocessing.postprocessing_options import InpaintingWhen


class FaceSwapUnit(BaseModel) :
    
    # The image given in reference
    source_img: str = Field(description='base64 reference image', examples=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQECWAJYAAD...."], default=None)
    # The checkpoint file
    source_face : str = Field(description='face checkpoint (from models/roop/faces)',examples=["my_face.pkl"], default=None)
    # base64 batch source images
    batch_images: Tuple[str] = Field(description='list of base64 batch source images',examples=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQECWAJYAAD....", "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQECWAJYAAD...."], default=None)

    # Will blend faces if True
    blend_faces: bool = Field(description='Will blend faces if True', default=True)
    
    # Use same gender filtering
    same_gender: bool = Field(description='Use same gender filtering', default=True)

    # If True, discard images with low similarity
    check_similarity : bool = Field(description='If True, discard images with low similarity', default=False) 
    # if True will compute similarity and add it to the image info
    compute_similarity : bool = Field(description='If True will compute similarity and add it to the image info', default=False) 

    # Minimum similarity against the used face (reference, batch or checkpoint)
    min_sim: float = Field(description='Minimum similarity against the used face (reference, batch or checkpoint)', default=0.0) 
    # Minimum similarity against the reference (reference or checkpoint if checkpoint is given)
    min_ref_sim: float = Field(description='Minimum similarity against the reference (reference or checkpoint if checkpoint is given)', default=0.0)

    # The face index to use for swapping
    faces_index: Tuple[int] = Field(description='The face index to use for swapping, list of face numbers starting from 0', default=(0,))

    def get_batch_images(self) -> List[Image.Image] :
        images = []
        if self.batch_images :
            for img in self.batch_images :
                images.append(base64_to_pil(img))
        return images

class PostProcessingOptions (BaseModel):
    face_restorer_name: str = Field(description='face restorer name', default=None)
    restorer_visibility: float = Field(description='face restorer visibility', default=1, le=1, ge=0)
    codeformer_weight: float = Field(description='face restorer codeformer weight', default=1, le=1, ge=0)

    upscaler_name: str = Field(description='upscaler name', default=None)
    scale: float = Field(description='upscaling scale', default=1, le=10, ge=0)
    upscale_visibility: float = Field(description='upscaler visibility', default=1, le=1, ge=0)
    
    inpainting_denoising_strengh : float = Field(description='Inpainting denoising strenght', default=0.02, lt=1, ge=0)
    inpainting_prompt : str = Field(description='Inpainting denoising strenght',examples=["Portrait of a [gender]"], default="Portrait of a [gender]")
    inpainting_negative_prompt : str =  Field(description='Inpainting denoising strenght',examples=["Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation"], default="")
    inpainting_steps : int = Field(description='Inpainting steps',examples=["Portrait of a [gender]"], ge=1, le=150)
    inpainting_sampler : str = Field(description='Inpainting sampler',examples=["Euler"], default="Euler")
    inpainting_when : InpaintingWhen = Field(description='When inpainting happens', examples=[e.value for e in InpaintingWhen.__members__.values()], default=InpaintingWhen.BEFORE_UPSCALING)


class FaceSwapRequest(BaseModel) : 
    image : str = Field(description='base64 reference image', examples=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQECWAJYAAD...."], default=None)
    units : List[FaceSwapUnit]
    postprocessing : PostProcessingOptions