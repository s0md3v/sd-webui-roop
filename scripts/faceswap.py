import importlib
from scripts.roop_utils.models_utils import get_models, get_face_checkpoints

from scripts import (roop_globals, roop_logging, faceswap_settings, faceswap_tab, faceswap_unit_ui)
from scripts.roop_swapping import swapper
from scripts.roop_utils import imgutils
from scripts.roop_utils import models_utils
from scripts.roop_postprocessing import upscaling
from pprint import pprint
import numpy as np
import logging

#Reload all the modules when using "apply and restart"
#This is mainly done for development purposes
importlib.reload(swapper)
importlib.reload(roop_logging)
importlib.reload(roop_globals)
importlib.reload(imgutils)
importlib.reload(upscaling)
importlib.reload(faceswap_settings)
importlib.reload(models_utils)
importlib.reload(faceswap_unit_ui)

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
from insightface.app.common import Face
from modules import scripts, shared
from modules.images import save_image, image_grid
from modules.processing import (Processed, StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img)
from modules.shared import opts
from PIL import Image

from scripts.roop_utils.imgutils import (pil_to_cv2,convert_to_sd)

from scripts.roop_logging import logger, save_img_debug
from scripts.roop_globals import VERSION_FLAG
from scripts.roop_postprocessing.postprocessing_options import PostProcessingOptions
from scripts.roop_postprocessing.postprocessing import enhance_image
from scripts.faceswap_unit_settings import FaceSwapUnitSettings


EXTENSION_PATH=os.path.join("extensions","sd-webui-roop")


# Register the tab, done here to prevent it from being added twice
script_callbacks.on_ui_tabs(faceswap_tab.on_ui_tabs)


class FaceSwapScript(scripts.Script):

    def __init__(self) -> None:
        logger.info(f"Roop {VERSION_FLAG}")
        super().__init__()

    @property
    def units_count(self) :
        return opts.data.get("roop_units_count", 3)
    
    @property
    def upscaled_swapper_in_generated(self) :
        return opts.data.get("roop_upscaled_swapper", False)
    
    @property
    def upscaled_swapper_in_source(self) :
        return opts.data.get("roop_upscaled_swapper_in_source", False)
    
    @property
    def enabled(self) -> bool :
        """Return True if any unit is enabled and the state is not interupted"""
        return any([u.enable for u in self.units]) and not shared.state.interrupted

    @property
    def model(self) -> str :
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


    def ui(self, is_img2img):
        with gr.Accordion(f"Roop {VERSION_FLAG}", open=False):
            components = []
            for i in range(1, self.units_count + 1):
                components += faceswap_unit_ui.faceswap_unit_ui(is_img2img, i)
            upscaler = faceswap_tab.upscaler_ui()
        # If the order is modified, the before_process should be changed accordingly.
        return components + upscaler

    # def make_script_first(self,p: StableDiffusionProcessing) :
    # FIXME : not really useful, will only impact postprocessing (kept for further testing)
    #     runner : scripts.ScriptRunner = p.scripts
    #     alwayson = runner.alwayson_scripts
    #     alwayson.pop(alwayson.index(self))
    #     alwayson.insert(0, self)
    #     print("Running in ", alwayson.index(self), "position")
    #     logger.info("Running scripts : %s", pformat(runner.alwayson_scripts))

    def read_config(self, p : StableDiffusionProcessing, *components) :
        # The order of processing for the components is important
        # The method first process faceswap units then postprocessing units 

        # self.make_first_script(p)

        self.units: List[FaceSwapUnitSettings] = []
        
        #Parse and convert units flat components into FaceSwapUnitSettings
        for i in range(0, self.units_count):
            self.units += [FaceSwapUnitSettings.get_unit_configuration(i, components)]

        for i, u in enumerate(self.units):
            logger.debug("%s, %s", pformat(i), pformat(u))

        #Parse the postprocessing options
        #We must first find where to start from (after face swapping units) 
        len_conf: int = len(fields(FaceSwapUnitSettings))
        shift: int = self.units_count * len_conf
        self.postprocess_options = PostProcessingOptions(
            *components[shift : shift + len(fields(PostProcessingOptions))]
        )
        logger.debug("%s", pformat(self.postprocess_options))

        if self.enabled :
            p.do_not_save_samples = not self.keep_original_images


    def process(self, p: StableDiffusionProcessing, *components):

        self.read_config(p, *components)

        #If is instance of img2img, we check if face swapping in source is required.
        if isinstance(p, StableDiffusionProcessingImg2Img):
            if self.enabled:
                init_images = p.init_images
                for i, unit in enumerate(self.units):
                    if unit.enable and unit.swap_in_source :
                        blend_config = unit.blend_faces # store blend config
                        unit.blend_faces = True # force blending
                        (init_images, result_infos) = self.process_images_unit(unit, init_images, upscaled_swapper=self.upscaled_swapper_in_source)
                        logger.info(f"unit {i+1}> processed init images: {len(init_images)}, {len(result_infos)}")
                        unit.blend_faces = blend_config #restore blend config

                p.init_images = init_images                        

    def process_image_unit(self, unit : FaceSwapUnitSettings, image: Image.Image, info = None, upscaled_swapper = False) -> List:
        """Process one image and return a List of (image, info) (one if blended, many if not).

        Args:
            unit : the current unit
            image : the image where to apply swapping
            info : The info

        Returns:
            List of tuple of (image, info) where image is the image where swapping has been applied and info is the image info with similarity infos.
        """

        results = []
        if unit.enable :
            if convert_to_sd(image) :
                return [(image, info)]
            if not unit.blend_faces :
                src_faces = unit.faces
                logger.info(f"will generate {len(src_faces)} images")
            else :
                logger.info("blend all faces together")
                src_faces = [unit.blended_faces]
                assert(not np.array_equal(unit.reference_face.embedding,src_faces[0].embedding) if len(unit.faces)>1 else True), "Reference face cannot be the same as blended"


            for i,src_face in enumerate(src_faces):
                logger.info(f"Process face {i}")
                if unit.reference_face is not None :
                    reference_face = unit.reference_face
                else :
                    logger.info("Use source face as reference face")
                    reference_face = src_face

                save_img_debug(image, "Before swap")
                result: swapper.ImageResult = swapper.swap_face(
                    reference_face,
                    src_face,
                    image,
                    faces_index=unit.faces_index,
                    model=self.model,
                    same_gender=unit.same_gender,
                    upscaled_swapper=upscaled_swapper,
                    compute_similarity=unit.compute_similarity
                )
                save_img_debug(result.image, "After swap")

                if result.image is None :
                    logger.error("Result image is None")
                if (not unit.check_similarity) or result.similarity and all([result.similarity.values()!=0]+[x >= unit.min_sim for x in result.similarity.values()]) and all([result.ref_similarity.values()!=0]+[x >= unit.min_ref_sim for x in result.ref_similarity.values()]):
                    results.append((result.image, f"{info}, similarity = {result.similarity}, ref_similarity = {result.ref_similarity}"))
                else:
                    logger.warning(
                        f"skip, similarity to low, sim = {result.similarity} (target {unit.min_sim}) ref sim = {result.ref_similarity} (target = {unit.min_ref_sim})"
                    )
        return results

    def process_images_unit(self, unit : FaceSwapUnitSettings, images : List[Image.Image], infos = None, upscaled_swapper = False) -> Tuple[List[Image.Image], List[str]] :
        if unit.enable :
            result_images : List[Image.Image] = []
            result_infos : List[str]= []
            if not infos :
                # this allows the use of zip afterwards if no infos are present
                # we make sure infos size is the same as images size
                infos = [None] * len(images)
            for i, (img, info) in enumerate(zip(images, infos)):
                swapped_images = self.process_image_unit(unit, img, info, upscaled_swapper)
                for result_image, result_info in swapped_images :
                    result_images.append(result_image)
                    result_infos.append(result_info)
            logger.info(f"{len(result_images)} images processed")
            return (result_images, result_infos)
        return (images, infos)

    def postprocess(self, p : StableDiffusionProcessing, processed: Processed, *args):
        if self.enabled :
            # Get the original images without the grid
            orig_images = processed.images[processed.index_of_first_image:]
            orig_infotexts = processed.infotexts[processed.index_of_first_image:]

            # These are were images and infos of swapped images will be stored
            images = []
            infotexts = []

            for i,(img,info) in enumerate(zip(orig_images, orig_infotexts)): 
                if any([u.enable for u in self.units]):
                    for unit_i, unit in enumerate(self.units):
                        #convert image position to batch index
                        #this should work (not completely confident)
                        batch_index = i%p.batch_size
                        if unit.enable :
                            if unit.swap_in_generated :
                                img_to_swap = img
                                
                                swapped_images = self.process_image_unit(image=img_to_swap, unit=unit, info=info, upscaled_swapper=self.upscaled_swapper_in_generated)
                                logger.info(f"{len(swapped_images)} images swapped")
                                for swp_img, new_info in swapped_images :
                                    logger.info(f"unit {unit_i+1}> processed")
                                    if swp_img is not None :

                                        save_img_debug(swp_img,"Before apply mask")
                                        swp_img = imgutils.apply_mask(swp_img, p, batch_index)
                                        save_img_debug(swp_img,"After apply mask")

                                        try :   
                                            if self.postprocess_options is not None:
                                                swp_img = enhance_image(swp_img, self.postprocess_options)
                                        except Exception as e:
                                            logger.error("Failed to upscale : %s", e)

                                        logger.info("Add swp image to processed")
                                        images.append(swp_img)
                                        infotexts.append(new_info)
                                        if p.outpath_samples and opts.samples_save :
                                            save_image(swp_img, p.outpath_samples, "", p.all_seeds[batch_index], p.all_prompts[batch_index], opts.samples_format,info=new_info, p=p, suffix="-swapped")      
                                    else :
                                        logger.error("swp image is None")
                            elif unit.swap_in_source and not self.keep_original_images :
                                # if images were swapped in source, but we don't keep original
                                # no images will be showned unless we add it as a swap image :
                                images.append(img)
                                infotexts.append(new_info)


            # Generate grid :
            if opts.return_grid and len(images) > 1:
                # FIXME :Use sd method, not that if blended is not active, the result will be a bit messy.
                grid = imgutils.create_square_image(images)
                text = processed.infotexts[0]
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                images.insert(0, grid)

            if self.keep_original_images:
                # If we want to keep original images, we add all existing (including grid this time)
                images += processed.images
                infotexts += processed.infotexts

            processed.images = images
            processed.infotexts = infotexts