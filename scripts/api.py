from PIL import Image
import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from modules.api.models import *
from modules.api import api
from scripts.roop_globals import VERSION_FLAG
import gradio as gr

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
    
    @app.get("/roop/swap_face")
    async def swap_face():
        return {"version": VERSION_FLAG}

try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(roop_api)
except:

    pass
