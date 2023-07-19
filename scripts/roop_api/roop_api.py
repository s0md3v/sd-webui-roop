from PIL import Image
import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from modules.api.models import *
from modules.api import api
from scripts.roop_api.roop_api_types import FaceSwapUnit, FaceSwapRequest
from scripts.roop_globals import VERSION_FLAG
import gradio as gr
from typing import Dict, List, Set, Tuple, Union, Optional
import json
from scripts.roop_swapping import swapper
from scripts.faceswap_unit_settings import FaceSwapUnitSettings
from scripts.roop_utils.imgutils import (pil_to_cv2,convert_to_sd, base64_to_pil)
from scripts.roop_utils.models_utils import get_current_model

def encode_to_base64(image):
    if type(image) is str:
        return image
    elif type(image) is Image.Image:
        return api.encode_pil_to_base64(image)
    elif type(image) is np.ndarray:
        return encode_np_to_base64(image)
    else:
        return ""

def encode_np_to_base64(image):
    pil = Image.fromarray(image)
    return api.encode_pil_to_base64(pil)


def roop_api(_: gr.Blocks, app: FastAPI):
    @app.get("/roop/version")
    async def version():
        return {"version": VERSION_FLAG}
    
    # use post as we consider the method non idempotent (which is debatable)
    @app.post("/roop/swap_face")
    async def swap_face(request : FaceSwapRequest) -> List[FaceSwapUnit]:
        units : List[FaceSwapUnitSettings]= []
        src_image = base64_to_pil(request.image)
        for u in request.units:
            units.append(
                FaceSwapUnitSettings(source_img=base64_to_pil(u.source_img),
                                source_face = u.source_face,
                                _batch_files = u.get_batch_images(),
                                blend_faces= u.blend_faces,
                                enable = True,
                                same_gender = u.same_gender,
                                check_similarity=u.check_similarity,
                                _compute_similarity=u.compute_similarity,
                                min_ref_sim= u.min_ref_sim,
                                min_sim= u.min_sim,
                                _faces_index = ",".join([str(i) for i in (u.faces_index)]),
                                swap_in_generated=True,
                                swap_in_source=False
                            ) 
            )


        # result_images = []
        # for unit_i, unit in enumerate(units):
        #         swapped_images = swapper.process_image_unit(get_current_model(), image=, unit=unit, info=info, upscaled_swapper=self.upscaled_swapper_in_generated)



        return request.units

