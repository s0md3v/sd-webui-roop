import importlib
from modules.scripts import PostprocessImageArgs,scripts_postprocessing
from scripts.roop_utils.models_utils import get_models, get_face_checkpoints

from scripts import (roop_globals, roop_logging, faceswap_settings, faceswap_tab)
from scripts.roop_swapping import swapper
from scripts.roop_utils import imgutils
from scripts.roop_utils import models_utils
from scripts.roop_postprocessing import upscaling


#Reload all the modules when using "apply and restart"
importlib.reload(swapper)
importlib.reload(roop_logging)
importlib.reload(roop_globals)
importlib.reload(imgutils)
importlib.reload(upscaling)
importlib.reload(faceswap_settings)
importlib.reload(models_utils)

import base64
import io
import os
from dataclasses import dataclass, fields
from pprint import pformat
from typing import Dict, List, Set, Tuple, Union, Optional

import dill as pickle
import gradio as gr
import modules.scripts as scripts
from modules import script_callbacks, scripts
import torch
from insightface.app.common import Face
from modules import processing, scripts, shared
from modules.images import save_image, image_grid
from modules.processing import (Processed, StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img)
from modules.shared import cmd_opts, opts, state
from PIL import Image

from scripts.roop_utils.imgutils import (pil_to_cv2,convert_to_sd)

from scripts.roop_logging import logger
from scripts.roop_globals import VERSION_FLAG
from scripts.roop_postprocessing.postprocessing_options import PostProcessingOptions
from scripts.roop_postprocessing.postprocessing import enhance_image


import modules

EXTENSION_PATH=os.path.join("extensions","sd-webui-roop")

@dataclass
class FaceSwapUnitSettings:
    # The image given in reference
    source_img: Union[Image.Image, str]
    # The checkpoint file
    source_face : str
    # The batch source images
    _batch_files: gr.components.File
    # Will blend faces if True
    blend_faces: bool
    # Enable this unit
    enable: bool
    # Use same gender filtering
    same_gender: bool

    # If True, discard images with low similarity
    check_similarity : bool 

    # Minimum similarity against the used face (reference, batch or checkpoint)
    min_sim: float
    # Minimum similarity against the reference (reference or checkpoint if checkpoint is given)
    min_ref_sim: float
    # The face index to use for swapping
    _faces_index: int
    # Swap in the source image in img2img (before processing)
    swap_in_source: bool
    # Swap in the generated image in img2img (always on for txt2img)
    swap_in_generated: bool

    @staticmethod
    def get_unit_configuration(unit: int, components):
        fields_count = len(fields(FaceSwapUnitSettings))
        return FaceSwapUnitSettings(
            *components[unit * fields_count : unit * fields_count + fields_count]
        )

    @property
    def faces_index(self):
        """
        Convert _faces_index from str to int 
        """
        faces_index = {
            int(x) for x in self._faces_index.strip(",").split(",") if x.isnumeric()
        }
        if len(faces_index) == 0:
            return {0}

        return faces_index

    @property
    def batch_files(self):
        """
        Return empty array instead of None for batch files
        """
        return self._batch_files or []
    
    @property
    def reference_face(self) :
        """
        Extract reference face (only once and store it for the rest of processing).
        Reference face is the checkpoint or the source image or the first image in the batch in that order.
        """
        if not hasattr(self,"_reference_face") :
            if self.source_face and self.source_face != "None" :
                with open(self.source_face, "rb") as file:
                    try :
                        logger.info(f"loading pickle {file.name}")
                        face = Face(pickle.load(file))
                        self._reference_face = face
                    except Exception as e :
                        logger.error("Failed to load checkpoint  : %s", e)
            elif self.source_img is not None :
                if isinstance(self.source_img, str):  # source_img is a base64 string
                    if 'base64,' in self.source_img:  # check if the base64 string has a data URL scheme
                        base64_data = self.source_img.split('base64,')[-1]
                        img_bytes = base64.b64decode(base64_data)
                    else:
                        # if no data URL scheme, just decode
                        img_bytes = base64.b64decode(self.source_img)
                    self.source_img = Image.open(io.BytesIO(img_bytes))
                source_img = pil_to_cv2(self.source_img)
                self._reference_face =  swapper.get_or_default(swapper.get_faces(source_img), 0, None)    
            else :
                logger.error("You need at least one face")
                self._reference_face = None
                
        return self._reference_face
    
    @property
    def faces(self) :
        """_summary_
        Extract all faces (including reference face) to provide an array of faces
        Only processed once.
        """
        if self.batch_files is not None and not hasattr(self,"_faces") :
            self._faces = [self.reference_face] if self.reference_face is not None else []
            for file in self.batch_files :
                img = Image.open(file.name)
                face = swapper.get_or_default(swapper.get_faces(pil_to_cv2(img)), 0, None)
                if face is not None :
                    self._faces.append(face)
        return self._faces

    @property
    def blended_faces(self):
        """
        Blend the faces using the mean of all embeddings
        """
        if not hasattr(self,"_blended_faces") :
            self._blended_faces = swapper.blend_faces(self.faces)
        return self._blended_faces


script_callbacks.on_ui_tabs(faceswap_tab.on_ui_tabs)


class FaceSwapScript(scripts.Script):

    @property
    def units_count(self) :
        return opts.data.get("roop_units_count", 3)
    
    @property
    def upscaled_swapper(self) :
        return opts.data.get("roop_upscaled_swapper", False)

    @property
    def enabled(self) :
        return any([u.enable for u in self.units]) and not shared.state.interrupted

    @property
    def model(self) :
        model = opts.data.get("roop_model", None)
        if model is None :
            models = get_models()
            model = models[0] if len(models) else None
        logger.info("Try to use model : %s", model)
        if not os.path.isfile(model):
            logger.error("The model %s cannot be found or loaded", model)
            raise FileNotFoundError("No faceswap model found. Please add it to the roop directory.")
        return model

    @property
    def keep_original_images(self) :
        return opts.data.get("roop_keep_original", False)

    def title(self):
        return f"roop"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def faceswap_unit_ui(self, is_img2img, unit_num=1):
        with gr.Tab(f"Face {unit_num}"):
            with gr.Column():
                gr.Markdown(
                """Reference is an image. First face will be extracted. 
                First face of batches sources will be extracted and used as input (or blended if blend is activated).""")
                with gr.Row():
                    img = gr.components.Image(type="pil", label="Reference")
                    batch_files = gr.components.File(
                        type="file",
                        file_count="multiple",
                        label="Batch Sources Images",
                        optional=True,
                    )
                gr.Markdown(
                    """Face checkpoint built with the checkpoint builder in tools. Will overwrite reference image.""")     
                with gr.Row() :
               
                    face = gr.inputs.Dropdown(
                        choices=get_face_checkpoints(),
                        label="Face Checkpoint (precedence over reference face)",
                    )
                    refresh = gr.Button(value='↻', variant='tool')
                    def refresh_fn(selected):
                        return gr.Dropdown.update(value=selected, choices=get_face_checkpoints())
                    refresh.click(fn=refresh_fn,inputs=face, outputs=face)

                with gr.Row():
                    enable = gr.Checkbox(False, placeholder="enable", label="Enable")
                    same_gender = gr.Checkbox(
                        False, placeholder="Same Gender", label="Same Gender"
                    )
                    blend_faces = gr.Checkbox(
                        True, placeholder="Blend Faces", label="Blend Faces ((Source|Checkpoint)+References = 1)"
                    )
                gr.Markdown("""Discard images with low similarity or no faces :""")
                check_similarity = gr.Checkbox(False, placeholder="discard", label="Check similarity")        
                min_sim = gr.Slider(0, 1, 0, step=0.01, label="Min similarity")
                min_ref_sim = gr.Slider(
                    0, 1, 0, step=0.01, label="Min reference similarity"
                )
                faces_index = gr.Textbox(
                    value="0",
                    placeholder="Which face to swap (comma separated), start from 0 (by gender if same_gender is enabled)",
                    label="Comma separated face number(s)",
                )
                gr.Markdown("""Configure swapping. Swapping can occure before img2img, after or both :""", visible=is_img2img)        
                swap_in_source = gr.Checkbox(
                    False,
                    placeholder="Swap face in source image",
                    label="Swap in source image (must be blended)",
                    visible=is_img2img,
                )
                swap_in_generated = gr.Checkbox(
                    True,
                    placeholder="Swap face in generated image",
                    label="Swap in generated image",
                    visible=is_img2img,
                )
        return [
            img,
            face,
            batch_files,
            blend_faces,
            enable,
            same_gender,
            check_similarity,
            min_sim,
            min_ref_sim,
            faces_index,
            swap_in_source,
            swap_in_generated,
        ]

    def ui(self, is_img2img):
        with gr.Accordion(f"Roop {VERSION_FLAG}", open=False):
            components = []
            for i in range(1, self.units_count + 1):
                components += self.faceswap_unit_ui(is_img2img, i)
            upscaler = faceswap_tab.upscaler_ui()
        return components + upscaler

    def before_process(self, p: StableDiffusionProcessing, *components):
        self.units: List[FaceSwapUnitSettings] = []
        for i in range(0, self.units_count):
            self.units += [FaceSwapUnitSettings.get_unit_configuration(i, components)]

        for i, u in enumerate(self.units):
            logger.debug("%s, %s", pformat(i), pformat(u))

        len_conf: int = len(fields(FaceSwapUnitSettings))
        shift: int = self.units_count * len_conf
        self.postprocess_options = PostProcessingOptions(
            *components[shift : shift + len(fields(PostProcessingOptions))]
        )
        logger.debug("%s", pformat(self.postprocess_options))


        if isinstance(p, StableDiffusionProcessingImg2Img):
            if any([u.enable for u in self.units]):
                init_images = p.init_images
                for i, unit in enumerate(self.units):
                    if unit.enable and unit.swap_in_source :
                        (init_images, result_infos) = self.process_images_unit(unit, init_images)
                        logger.info(f"unit {i+1}> processed init images: {len(init_images)}, {len(result_infos)}")

                p.init_images = init_images


    def postprocess_batch(self, p, *args, **kwargs):
        if self.enabled :
            if self.keep_original_images:
                batch_index = kwargs.pop('batch_number', 0)
                torch_images : torch.Tensor = kwargs["images"]
                pil_images = imgutils.torch_to_pil(torch_images)
                self._orig_images = pil_images
                for img in pil_images :
                    if p.outpath_samples and opts.samples_save :
                        save_image(img, p.outpath_samples, "", p.seeds[batch_index], p.prompts[batch_index], opts.samples_format, p=p, suffix="-before-swap")

                return 

    def process_image_unit(self, unit : FaceSwapUnitSettings, image, info = None) -> Tuple[Optional[Image.Image], Optional[str]]:
        if unit.enable :
            if convert_to_sd(image) :
                return (image, info)
            if not unit.blend_faces :
                src_faces = unit.faces
                logger.info(f"will generate {len(src_faces)} images")
            else :
                logger.info("blend all faces together")
                src_faces = [unit.blended_faces]
            for i,src_face in enumerate(src_faces):
                logger.info(f"Process face {i}")
                result: swapper.ImageResult = swapper.swap_face(
                    unit.reference_face if unit.reference_face is not None else src_face,
                    src_face,
                    image,
                    faces_index=unit.faces_index,
                    model=self.model,
                    same_gender=unit.same_gender,
                    upscaled_swapper=self.upscaled_swapper
                )
                if (not unit.check_similarity) or result.similarity and all([result.similarity.values()!=0]+[x >= unit.min_sim for x in result.similarity.values()]) and all([result.ref_similarity.values()!=0]+[x >= unit.min_ref_sim for x in result.ref_similarity.values()]):
                    return (result.image, f"{info}, similarity = {result.similarity}, ref_similarity = {result.ref_similarity}")
                else:
                    logger.warning(
                        f"skip, similarity to low, sim = {result.similarity} (target {unit.min_sim}) ref sim = {result.ref_similarity} (target = {unit.min_ref_sim})"
                    )
        return (None, None)


    def process_images_unit(self, unit : FaceSwapUnitSettings, images : List[Image.Image], infos = None) -> Tuple[List[Image.Image], List[str]] :
        if unit.enable :
            result_images : List[Image.Image] = []
            result_infos : List[str]= []
            if not infos :
                infos = [None] * len(images)
            for i, (img, info) in enumerate(zip(images, infos)):
                (result_image, result_info) = self.process_image_unit(unit, img, info)
                if result_image is not None and result_info is not None :
                    result_images.append(result_image)
                    result_infos.append(result_info)
            logger.info(f"{len(result_images)} images processed")
            return (result_images, result_infos)
        return (images, infos)

    def postprocess_image(self, p, script_pp: PostprocessImageArgs, *args):
        if self.enabled :
            img : Image.Image = script_pp.image
            infos = ""
            if any([u.enable for u in self.units]):
                for i, unit in enumerate(self.units):
                    if unit.enable :
                        img,info = self.process_image_unit(image=img, unit=unit, info="")
                        logger.info(f"unit {i+1}> processed")
                        infos += info or ""
                        if img is None :
                            logger.error("Failed to process image - Switch back to original image")
                            img = script_pp.image
            try :   
                if self.postprocess_options is not None:
                    img = enhance_image(img, self.postprocess_options)
            except Exception as e:
                logger.error("Failed to upscale : %s", e)
            pp = scripts_postprocessing.PostprocessedImage(img)
            pp.info = {"face.similarity" : infos}
            p.extra_generation_params.update(pp.info)
            script_pp.image = pp.image

    def postprocess(self, p : StableDiffusionProcessing, processed: Processed, *args):
        if self.enabled :

            images = processed.images[processed.index_of_first_image:]
            for i,img in enumerate(images) :
                images[i] = processing.apply_overlay(img, p.paste_to, i%p.batch_size, p.overlay_images)

            processed.images = images

            if self.keep_original_images:
                if len(self._orig_images)> 1 :
                    processed.images.append(image_grid(self._orig_images))
                processed.images += self._orig_images
                processed.infotexts+= processed.infotexts # duplicate infotexts           


