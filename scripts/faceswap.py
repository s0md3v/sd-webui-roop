import glob
import importlib

from scripts import (cimage, imgutils, roop_logging, roop_version, swapper,
                     upscaling)

#Reload all the modules when using "apply and restart"
importlib.reload(swapper)
importlib.reload(roop_logging)
importlib.reload(roop_version)
importlib.reload(cimage)
importlib.reload(imgutils)
importlib.reload(upscaling)

import base64
import io
import json
import os
import tempfile
from dataclasses import dataclass, fields
from pprint import pformat, pprint
from typing import Dict, List, Set, Tuple, Union

import cv2
import dill as pickle
import gradio as gr
import modules.scripts as scripts
import numpy as np
import onnx
import pandas as pd
import torch
from insightface.app.common import Face
from modules import processing, script_callbacks, scripts, shared
from modules.face_restoration import FaceRestoration
from modules.images import save_image
from modules.processing import (Processed, StableDiffusionProcessing,
                                StableDiffusionProcessingImg2Img,
                                StableDiffusionProcessingTxt2Img)
from modules.shared import cmd_opts, opts, state
from modules.upscaler import Upscaler, UpscalerData
from onnx import numpy_helper
from PIL import Image

from scripts.cimage import convert_to_sd
from scripts.imgutils import (create_square_image, cv2_to_pil, pil_to_cv2,
                              pil_to_torch, torch_to_pil)
from scripts.roop_logging import logger
from scripts.roop_version import version_flag
from scripts.upscaling import UpscaleOptions, upscale_image

import modules

EXTENSION_PATH=os.path.join("extensions","sd-webui-roop")

def get_models():
    """
    Retrieve a list of swap model files.

    This function searches for model files in the specified directories and returns a list of file paths.
    The supported file extensions are ".onnx".

    Returns:
        A list of file paths of the model files.
    """
    models_path = os.path.join(scripts.basedir(), EXTENSION_PATH, "models", "*")
    models = glob.glob(models_path)

    # Add an additional models directory and find files in it
    models_path = os.path.join(scripts.basedir(), "models", "roop", "*")
    models += glob.glob(models_path)

    # Filter the list to include only files with the supported extensions
    models = [x for x in models if x.endswith(".onnx")]

    return models

def get_face_checkpoints():
    """
    Retrieve a list of face checkpoint paths.

    This function searches for face files with the extension ".pkl" in the specified directory and returns a list
    containing the paths of those files.

    Returns:
        list: A list of face paths, including the string "None" as the first element.
    """
    faces_path = os.path.join(scripts.basedir(), "models", "roop", "faces", "*.pkl")
    faces = glob.glob(faces_path)
    return ["None"] + faces

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


def compare(img1, img2):
    if img1 is not None and img2 is not None:
        return swapper.compare_faces(img1, img2)

    return "You need 2 images to compare"



def extract_faces(files, extract_path,  face_restorer_name, face_restorer_visibility,upscaler_name,upscaler_scale, upscaler_visibility,inpainting_denoising_strengh, inpainting_prompt, inpainting_negative_prompt, inpainting_steps):
    if not extract_path :
        tempfile.mkdtemp()
    if files is not None:
        images = []
        for file in files :
            img = Image.open(file.name).convert("RGB")
            faces = swapper.get_faces(pil_to_cv2(img))
            if faces:
                face_images = []
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x_min, y_min, x_max, y_max = bbox
                    face_image = img.crop((x_min, y_min, x_max, y_max))
                    if face_restorer_name or face_restorer_visibility:
                        scale = 1 if face_image.width > 512 else 512//face_image.width
                        face_image = upscale_image(face_image, UpscaleOptions(face_restorer_name=face_restorer_name,
                                                                            restorer_visibility=face_restorer_visibility,
                                                                            upscaler_name=upscaler_name, 
                                                                            upscale_visibility=upscaler_visibility, 
                                                                            scale=scale,
                                                                            inpainting_denoising_strengh=inpainting_denoising_strengh,
                                                                            inpainting_prompt=inpainting_prompt,
                                                                            inpainting_steps=inpainting_steps,
                                                                            inpainting_negative_prompt=inpainting_negative_prompt))
                    path = tempfile.NamedTemporaryFile(delete=False,suffix=".png",dir=extract_path).name
                    face_image.save(path)
                    face_images.append(path)
                images+= face_images
        return images
    return None



def build_face_checkpoint_and_save(batch_files, name):
    """
    Builds a face checkpoint, swaps faces, and saves the result to a file.

    Args:
        batch_files (list): List of image file paths.
        name (str): Name of the face checkpoint

    Returns:
        PIL.Image.Image or None: Resulting swapped face image if successful, otherwise None.
    """
    batch_files = batch_files or []
    print("Build", name, [x.name for x in batch_files])
    faces = swapper.get_faces_from_img_files(batch_files)
    blended_face = swapper.blend_faces(faces)
    preview_path = os.path.join(
        scripts.basedir(), "extensions", "sd-webui-roop", "references"
    )
    faces_path = os.path.join(scripts.basedir(), "models", "roop","faces")

    target_img = None
    if blended_face:
        if blended_face["gender"] == 0:
            target_img = Image.open(os.path.join(preview_path, "woman.png"))
        else:
            target_img = Image.open(os.path.join(preview_path, "man.png"))

        if name == "":
            name = "default_name"
        pprint(blended_face)
        result = swapper.swap_face(blended_face, blended_face, target_img, get_models()[0])
        result_image = upscale_image(result.image, UpscaleOptions(face_restorer_name="CodeFormer", restorer_visibility=1))
        
        file_path = os.path.join(faces_path, f"{name}.pkl")
        file_number = 1
        while os.path.exists(file_path):
            file_path = os.path.join(faces_path, f"{name}_{file_number}.pkl")
            file_number += 1
        result_image.save(file_path+".png")
        with open(file_path, "wb") as file:
            pickle.dump({"embedding" :blended_face.embedding, "gender" :blended_face.gender, "age" :blended_face.age},file)
        try :
            with open(file_path, "rb") as file:
                data = Face(pickle.load(file))
                print(data)
        except Exception as e :
            print(e)
        return result_image

    print("No face found")

    return target_img




def explore_onnx_faceswap_model(model_path):
    data = {
        'Node Name': [],
        'Op Type': [],
        'Inputs': [],
        'Outputs': [],
        'Attributes': []
    }
    if model_path:
        model = onnx.load(model_path)
        for node in model.graph.node:
            data['Node Name'].append(pformat(node.name))
            data['Op Type'].append(pformat(node.op_type))
            data['Inputs'].append(pformat(node.input))
            data['Outputs'].append(pformat(node.output))
            attributes = []
            for attr in node.attribute:
                attr_name = attr.name
                attr_value = attr.t
                attributes.append("{} = {}".format(pformat(attr_name), pformat(attr_value)))
            data['Attributes'].append(attributes)

    df = pd.DataFrame(data)
    return df

def upscaler_ui():
    with gr.Tab(f"Enhancements"):
        gr.Markdown(
                """Upscaling is performed on the whole image. Upscaling happens before face restoration.""")
        with gr.Row():
            face_restorer_name = gr.Radio(
                label="Restore Face",
                choices=["None"] + [x.name() for x in shared.face_restorers],
                value=shared.face_restorers[0].name(),
                type="value",
            )
            face_restorer_visibility = gr.Slider(
                0, 1, 1, step=0.1, label="Restore visibility"
            )
        upscaler_name = gr.inputs.Dropdown(
            choices=[upscaler.name for upscaler in shared.sd_upscalers],
            label="Upscaler",
        )
        upscaler_scale = gr.Slider(1, 8, 1, step=0.1, label="Upscaler scale")
        upscaler_visibility = gr.Slider(
            0, 1, 1, step=0.1, label="Upscaler visibility (if scale = 1)"
        )
        with gr.Accordion(f"Post Inpainting (Beta)", open=True):
            gr.Markdown(
                """Inpainting sends image to inpainting with a mask on face (once for each faces).""")
            inpainting_when = gr.Dropdown(choices = [e.value for e in upscaling.InpaintingWhen.__members__.values()],value=[upscaling.InpaintingWhen.BEFORE_RESTORE_FACE.value], label="Enable/When")
            inpainting_denoising_strength = gr.Slider(
                0, 1, 0, step=0.01, label="Denoising strenght (will send face to img2img after processing)"
            )

            inpainting_denoising_prompt = gr.Textbox("Portrait of a [gender]", label="Inpainting prompt use [gender] instead of men or woman")
            inpainting_denoising_negative_prompt = gr.Textbox("", label="Inpainting negative prompt use [gender] instead of men or woman")
            with gr.Row():
                samplers_names = [s.name for s in modules.sd_samplers.all_samplers]
                inpainting_sampler = gr.Dropdown(
                        choices=samplers_names,
                        value=[samplers_names[0]],
                        label="Inpainting Sampler",
                    )
                inpainting_denoising_steps =  gr.Slider(
                    1, 150, 20, step=1, label="Inpainting steps"
                )
                
    return [
        face_restorer_name,
        face_restorer_visibility,
        upscaler_name,
        upscaler_scale,
        upscaler_visibility,
        inpainting_denoising_strength,
        inpainting_denoising_prompt,
        inpainting_denoising_negative_prompt,
        inpainting_denoising_steps,
        inpainting_sampler,
        inpainting_when
    ]

def tools_ui():
    models = get_models()
    with gr.Tab("Tools"):
        with gr.Tab("Build"):
            gr.Markdown(
                """Build a face based on a batch list of images. Will blend the resulting face and store the checkpoint in the roop/faces directory.""")
            with gr.Row():
                batch_files = gr.components.File(
                    type="file",
                    file_count="multiple",
                    label="Batch Sources Images",
                    optional=True,
                )
                preview = gr.components.Image(type="pil", label="Preview", interactive=False)
            name = gr.Textbox(
                value="Face",
                placeholder="Name of the character",
                label="Name of the character",
            )
            generate_checkpoint_btn = gr.Button("Save")
        with gr.Tab("Compare"):
            gr.Markdown(
                """Give a similarity score between two images (only first face is compared).""")
 
            with gr.Row():
                img1 = gr.components.Image(type="pil", label="Face 1")
                img2 = gr.components.Image(type="pil", label="Face 2")
            compare_btn = gr.Button("Compare")
            compare_result_text = gr.Textbox(
                interactive=False, label="Similarity", value="0"
            )
        with gr.Tab("Extract"):
            gr.Markdown(
                """Extract all faces from a batch of images. Will apply enhancement in the tools enhancement tab.""")
            with gr.Row():
                extracted_source_files = gr.components.File(
                    type="file",
                    file_count="multiple",
                    label="Batch Sources Images",
                    optional=True,
                )
                extracted_faces =  gr.Gallery(
                                        label="Extracted faces", show_label=False
                                    ).style(columns=[2], rows=[2])
            extract_save_path = gr.Textbox(label="Destination Directory", value="")
            extract_btn = gr.Button("Extract")
        with gr.Tab("Explore Model"):
            model = gr.inputs.Dropdown(
                choices=models,
                label="Model not found, please download one and reload automatic 1111",
            )            
            explore_btn = gr.Button("Explore")
            explore_result_text = gr.Dataframe(
                interactive=False, label="Explored"
            )
        upscale_options = upscaler_ui()

    explore_btn.click(explore_onnx_faceswap_model, inputs=[model], outputs=[explore_result_text])  
    compare_btn.click(compare, inputs=[img1, img2], outputs=[compare_result_text])
    generate_checkpoint_btn.click(build_face_checkpoint_and_save, inputs=[batch_files, name], outputs=[preview])
    extract_btn.click(extract_faces, inputs=[extracted_source_files, extract_save_path]+upscale_options, outputs=[extracted_faces])  

def on_ui_tabs() :
    with gr.Blocks(analytics_enabled=False) as ui_faceswap:
        tools_ui()
    return [(ui_faceswap, "Roop", "roop_tab")] 

script_callbacks.on_ui_tabs(on_ui_tabs)


def on_ui_settings():
    section = ('roop', "Roop")
    shared.opts.add_option("roop_units_count", shared.OptionInfo(
        3, "Max faces units (requires restart)", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section))
    shared.opts.add_option("roop_upscaled_swapper", shared.OptionInfo(
        False, "Upscaled swapper", gr.Checkbox, {"interactive": True}, section=section))
script_callbacks.on_ui_settings(on_ui_settings)


class FaceSwapScript(scripts.Script):

    @property
    def units_count(self) :
        return opts.data.get("roop_units_count", 3)
    
    @property
    def upscaled_swapper(self) :
        return opts.data.get("roop_upscaled_swapper", False)

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

    def configuration_ui(self, is_img2img):
        with gr.Tab(f"Settings"):
            models = get_models()
            show_unmodified = gr.Checkbox(
                False,
                placeholder="Show Unmodified",
                label="Show Unmodified (original)",
            )
            if len(models) == 0:
                logger.warning(
                    "You should at least have one model in models directory, please read the doc here : https://github.com/s0md3v/sd-webui-roop"
                )
                model = gr.inputs.Dropdown(
                    choices=models,
                    label="Model not found, please download one and reload automatic 1111",
                )
            else:
                model = gr.inputs.Dropdown(
                    choices=models, label="Model", default=models[0]
                )
        return [show_unmodified, model]

    def ui(self, is_img2img):
        with gr.Accordion(f"Roop {version_flag}", open=False):
            components = []
            for i in range(1, self.units_count + 1):
                components += self.faceswap_unit_ui(is_img2img, i)
            upscaler = upscaler_ui()
            configuration = self.configuration_ui(is_img2img)
            #tools_ui() #disable tools in accordion
        return components + upscaler + configuration

    def process(self, p: StableDiffusionProcessing, *components):
        self.units: List[FaceSwapUnitSettings] = []
        for i in range(0, self.units_count):
            self.units += [FaceSwapUnitSettings.get_unit_configuration(i, components)]

        for i, u in enumerate(self.units):
            logger.debug("%s, %s", pformat(i), pformat(u))

        len_conf: int = len(fields(FaceSwapUnitSettings))
        shift: int = self.units_count * len_conf
        self.upscale_options = UpscaleOptions(
            *components[shift : shift + len(fields(UpscaleOptions))]
        )
        logger.debug("%s", pformat(self.upscale_options))
        self.model = components[-1]
        self.show_unmodified = components[-2]

        if isinstance(p, StableDiffusionProcessingImg2Img):
            if any([u.enable for u in self.units]):
                init_images = p.init_images
                for i, unit in enumerate(self.units):
                    if unit.enable and unit.swap_in_source :
                        (init_images, result_infos) = self.process_images_unit(unit, init_images)
                        logger.info(f"unit {i+1}> processed init images: {len(init_images)}, {len(result_infos)}")

                p.init_images = init_images


    def process_images_unit(self, unit : FaceSwapUnitSettings, images, infos = None)  :
        if unit.enable :
            result_images = []
            result_infos = []
            if not infos :
                infos = [None] * len(images)
            for i, (img, info) in enumerate(zip(images, infos)):
                if convert_to_sd(img) :
                    result_infos.append(info)
                    result_images.append(img)
                    continue
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
                        img,
                        faces_index=unit.faces_index,
                        model=self.model,
                        same_gender=unit.same_gender,
                        upscaled_swapper=self.upscaled_swapper
                    )
                    if (not unit.check_similarity) or result.similarity and all([result.similarity.values()!=0]+[x >= unit.min_sim for x in result.similarity.values()]) and all([result.ref_similarity.values()!=0]+[x >= unit.min_ref_sim for x in result.ref_similarity.values()]):
                        result_infos.append(f"{info}, similarity = {result.similarity}, ref_similarity = {result.ref_similarity}")
                        result_images.append(result.image)
                    else:
                        logger.warning(
                            f"skip, similarity to low, sim = {result.similarity} (target {unit.min_sim}) ref sim = {result.ref_similarity} (target = {unit.min_ref_sim})"
                        )
            logger.info(f"{len(result_images)} images processed")
            return (result_images, result_infos)
        return (images, infos)

    def postprocess(self, p : StableDiffusionProcessing, processed: Processed, *args):
        orig_images = processed.images
        orig_infos = processed.infotexts

        if any([u.enable for u in self.units]):
            result_images = processed.images[:]
            result_infos = processed.infotexts[:]
            if p.batch_size > 1 or p.n_iter > 1:
                # Remove grid image if batch size is greater than 1 :
                result_images = result_images[1:]
                result_infos = result_infos[1:]
                logger.info("Discard grid image from swapping process. This could induce bugs with some extensions.")

            for i, unit in enumerate(self.units):
                if unit.enable and unit.swap_in_generated :
                    (result_images, result_infos) = self.process_images_unit(unit, result_images, result_infos)
                    logger.info(f"unit {i+1}> processed : {len(result_images)}, {len(result_infos)}")

            for i, img in enumerate(result_images):
                if self.upscale_options is not None:
                    result_images[i] = upscale_image(img, self.upscale_options)
                if p.outpath_samples and opts.samples_save :
                    save_image(result_images[i], p.outpath_samples, seed=int(p.seed), info=result_infos[i], basename="swapped")                           
            if len(result_images) > 1:
                try :
                    # prepend swapped grid to result_images :
                    result_images = [create_square_image(result_images)] + result_images
                except Exception as e :
                    logger.error("Error building result grid %s", e)
            processed.images = result_images
            processed.infotexts = result_infos
            
            if self.show_unmodified:
                processed.images += orig_images
                processed.infotexts+= orig_infos


