import importlib
from scripts.roop_utils.models_utils import get_models, get_face_checkpoints

from scripts import (roop_globals, roop_logging, faceswap_settings, faceswap_tab, faceswap_unit_ui)
from scripts.roop_swapping import swapper
from scripts.roop_utils import imgutils
from scripts.roop_utils import models_utils
from scripts.roop_postprocessing import upscaling
from pprint import pprint
import numpy as np

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

from scripts.roop_logging import logger
from scripts.roop_globals import VERSION_FLAG
from scripts.roop_postprocessing.postprocessing_options import PostProcessingOptions
from scripts.roop_postprocessing.postprocessing import enhance_image

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
    # if True will compute similarity and add it to the image info
    _compute_similarity :bool

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
    def compute_similarity(self) :
        return self._compute_similarity or self.check_similarity

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
                if self._reference_face is None :
                    logger.error("Face not found in reference image")  
            else :
                self._reference_face = None

        if self._reference_face is None :
            logger.error("You need at least one reference face")

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
            assert(all([not np.array_equal(self._blended_faces.embedding, face.embedding) for face in self.faces]) if len(self.faces) > 1 else True), "Blended faces cannot be the same as one of the face if len(face)>0"
            assert(not np.array_equal(self._blended_faces.embedding,self.reference_face.embedding)  if len(self.faces) > 1 else True), "Blended faces cannot be the same as reference face if len(face)>0"

        return self._blended_faces

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

    def before_process(self, p: StableDiffusionProcessing, *components):
        # The order of processing for the components is important
        # The method first process faceswap units then postprocessing units 

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

                # Apply mask :
                for i,img in enumerate(p.init_images) :
                    p.init_images[i] = imgutils.apply_mask(img, p, i)
                        


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

            if self.keep_original_images:
                # If we want to keep original images, we add all existing (including grid this time)
                images = processed.images
                infotexts = processed.infotexts

            for i,(img,info) in enumerate(zip(orig_images, orig_infotexts)): 
                if any([u.enable for u in self.units]):
                    for unit_i, unit in enumerate(self.units):
                        #convert image position to batch index
                        #this should work (not completely confident)
                        batch_index = i%p.batch_size
                        if unit.enable :
                            if unit.swap_in_generated :
                                swapped_images = self.process_image_unit(image=img, unit=unit, info=info, upscaled_swapper=self.upscaled_swapper_in_generated)
                                logger.info(f"{len(swapped_images)} images swapped")
                                for swp_img, new_info in swapped_images :
                                    logger.info(f"unit {unit_i+1}> processed")
                                    if swp_img is not None :
                                        swp_img = imgutils.apply_mask(swp_img, p, batch_index)
                                        try :   
                                            if self.postprocess_options is not None:
                                                swp_img = enhance_image(swp_img, self.postprocess_options)
                                        except Exception as e:
                                            logger.error("Failed to upscale : %s", e)

                                        logger.info("Add swp image to processed")
                                        images.append(swp_img)
                                        infotexts.append(new_info)
                                        if p.outpath_samples and opts.samples_save :
                                            save_image(swp_img, p.outpath_samples, "", p.seeds[batch_index], p.prompts[batch_index], opts.samples_format, p=p, suffix="-swapped")      
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
                grid = image_grid(images, p.batch_size)
                text = processed.infotexts[0]
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text
                images.insert(0, grid)


            processed.images = images
            processed.infotexts = infotexts