from typing import List, Tuple
from PIL import Image
from pydantic import BaseModel, Field
from enum import Enum
import base64, io
from io import BytesIO
from typing import Dict, List, Set, Tuple, Union, Optional

class InpaintingWhen(Enum):
    NEVER = "Never"
    BEFORE_UPSCALING = "Before Upscaling/all"
    BEFORE_RESTORE_FACE = "After Upscaling/Before Restore Face"
    AFTER_ALL = "After All"

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


class PostProcessingOptions (BaseModel):
    face_restorer_name: str = Field(description='face restorer name', default=None)
    restorer_visibility: float = Field(description='face restorer visibility', default=1, le=1, ge=0)
    codeformer_weight: float = Field(description='face restorer codeformer weight', default=1, le=1, ge=0)

    upscaler_name: str = Field(description='upscaler name', default=None)
    scale: float = Field(description='upscaling scale', default=1, le=10, ge=0)
    upscale_visibility: float = Field(description='upscaler visibility', default=1, le=1, ge=0)
    
    inpainting_denoising_strengh : float = Field(description='Inpainting denoising strenght', default=0, lt=1, ge=0)
    inpainting_prompt : str = Field(description='Inpainting denoising strenght',examples=["Portrait of a [gender]"], default="Portrait of a [gender]")
    inpainting_negative_prompt : str =  Field(description='Inpainting denoising strenght',examples=["Deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation"], default="")
    inpainting_steps : int = Field(description='Inpainting steps',examples=["Portrait of a [gender]"], ge=1, le=150, default=20)
    inpainting_sampler : str = Field(description='Inpainting sampler',examples=["Euler"], default="Euler")
    inpainting_when : InpaintingWhen = Field(description='When inpainting happens', examples=[e.value for e in InpaintingWhen.__members__.values()], default=InpaintingWhen.NEVER)


class FaceSwapRequest(BaseModel) : 
    image : str = Field(description='base64 reference image', examples=["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQECWAJYAAD...."], default=None)
    units : List[FaceSwapUnit]
    postprocessing : PostProcessingOptions


class FaceSwapResponse(BaseModel) :
    images : List[str] = Field(description='base64 swapped image',default=None)
    infos : List[str]

    @property
    def pil_images(self) :
        return [base64_to_pil(img) for img in self.images]

def pil_to_base64(img):
    if isinstance(img, str):
        img = Image.open(img)

    buffer = BytesIO()
    img.save(buffer, format='PNG')
    img_data = buffer.getvalue()
    base64_data = base64.b64encode(img_data)
    return base64_data.decode('utf-8')    

def base64_to_pil(base64str : Optional[str]) -> Optional[Image.Image] :
    if base64str is None :
        return None
    if 'base64,' in base64str:  # check if the base64 string has a data URL scheme
        base64_data = base64str.split('base64,')[-1]
        img_bytes = base64.b64decode(base64_data)
    else:
        # if no data URL scheme, just decode
        img_bytes = base64.b64decode(base64str)
    return Image.open(io.BytesIO(img_bytes))