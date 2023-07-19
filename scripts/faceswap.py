import importlib
from scripts.roop_api import roop_api
from scripts.roop_utils.models_utils import get_current_model, get_face_checkpoints

from scripts import (roop_globals, roop_logging, faceswap_settings, faceswap_tab, faceswap_unit_ui)
from scripts.roop_swapping import swapper
from scripts.roop_utils import imgutils
from scripts.roop_utils import models_utils
from scripts.roop_postprocessing import upscaling
from pprint import pprint
import numpy as np
import logging
from copy import deepcopy

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
importlib.reload(roop_api)

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

try:
    import modules.script_callbacks as script_callbacks
    script_callbacks.on_app_started(roop_api.roop_api)
except:
    pass


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
    def keep_original_images(self) :
        return opts.data.get("roop_keep_original", False)

    @property
    def swap_in_generated_units(self) :
        return [u for u in self.units if u.swap_in_generated and u.enable]

    @property
    def swap_in_source_units(self) :
        return [u for u in self.units if u.swap_in_source and u.enable]

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
            if self.enabled and len(self.swap_in_source_units) > 0:
                init_images : List[Tuple[Image.Image, str]] = [(img,None) for img in p.init_images]
                new_inits  = swapper.process_images_units(get_current_model(), self.swap_in_source_units,images=init_images, upscaled_swapper=self.upscaled_swapper_in_source,force_blend=True)
                logger.info(f"processed init images: {len(init_images)}")
                p.init_images = [img[0] for img in new_inits]                        


    def postprocess(self, p : StableDiffusionProcessing, processed: Processed, *args):
        if self.enabled :
            # Get the original images without the grid
            orig_images : List[Image.Image] = processed.images[processed.index_of_first_image:]
            orig_infotexts : List[str] = processed.infotexts[processed.index_of_first_image:]

            keep_original = self.keep_original_images

            # These are were images and infos of swapped images will be stored
            images = []
            infotexts = []
            if (len(self.swap_in_generated_units))>0 :
                for i,(img,info) in enumerate(zip(orig_images, orig_infotexts)): 
                    batch_index = i%p.batch_size
                    swapped_images = swapper.process_images_units(get_current_model(), self.swap_in_generated_units, images=[(img,info)], upscaled_swapper=self.upscaled_swapper_in_generated)
                    logger.info(f"{len(swapped_images)} images swapped")
                    
                    for swp_img, new_info in swapped_images :
                        img = swp_img # Will only swap the last image in the batch in next units (FIXME : hard to fix properly but not really critical)

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
            else :
                keep_original=True


            # Generate grid :
            if opts.return_grid and len(images) > 1:
                # FIXME :Use sd method, not that if blended is not active, the result will be a bit messy.
                grid = imgutils.create_square_image(images)
                text = processed.infotexts[0]
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                images.insert(0, grid)

            if keep_original:
                # If we want to keep original images, we add all existing (including grid this time)
                images += processed.images
                infotexts += processed.infotexts

            processed.images = images
            processed.infotexts = infotexts