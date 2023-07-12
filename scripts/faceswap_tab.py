import os
import tempfile
from pprint import pformat, pprint

import dill as pickle
import gradio as gr
import modules.scripts as scripts
import numpy as np
import onnx
import pandas as pd
from scripts.faceswap_upscaler_ui import upscaler_ui
from insightface.app.common import Face
from modules import script_callbacks, scripts
from PIL import Image

from scripts.roop_utils import imgutils
from scripts.roop_utils.imgutils import pil_to_cv2
from scripts.roop_utils.models_utils import get_models
from scripts.roop_logging import logger
import scripts.roop_swapping.swapper as swapper
from scripts.roop_postprocessing.postprocessing_options import PostProcessingOptions
from scripts.roop_postprocessing.postprocessing import enhance_image


def compare(img1, img2):
    if img1 is not None and img2 is not None:
        return swapper.compare_faces(img1, img2)

    return "You need 2 images to compare"



def extract_faces(files, extract_path,  face_restorer_name, face_restorer_visibility, codeformer_weight,upscaler_name,upscaler_scale, upscaler_visibility,inpainting_denoising_strengh, inpainting_prompt, inpainting_negative_prompt, inpainting_steps, inpainting_sampler,inpainting_when):
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
                        face_image = enhance_image(face_image, PostProcessingOptions(face_restorer_name=face_restorer_name,
                                                                            restorer_visibility=face_restorer_visibility,
                                                                            codeformer_weight= codeformer_weight,
                                                                            upscaler_name=upscaler_name, 
                                                                            upscale_visibility=upscaler_visibility, 
                                                                            scale=scale,
                                                                            inpainting_denoising_strengh=inpainting_denoising_strengh,
                                                                            inpainting_prompt=inpainting_prompt,
                                                                            inpainting_steps=inpainting_steps,
                                                                            inpainting_negative_prompt=inpainting_negative_prompt,
                                                                            inpainting_when=inpainting_when,
                                                                            inpainting_sampler=inpainting_sampler))
                    path = tempfile.NamedTemporaryFile(delete=False,suffix=".png",dir=extract_path).name
                    face_image.save(path)
                    face_images.append(path)
                images+= face_images
        return images
    return None

def analyse_faces(image, det_threshold = 0.5) :
    try :
        faces = swapper.get_faces(imgutils.pil_to_cv2(image), det_thresh=det_threshold)
        result = ""
        for i,face in enumerate(faces) :
            result+= f"\nFace {i} \n" + "="*40 +"\n"
            result+= pformat(face) + "\n"
            result+= "="*40
        return result

    except Exception as e :
        logger.error("Analysis Failed : %s", e)
        return "Analysis Failed"
    
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
    if not os.path.exists(faces_path):
        os.makedirs(faces_path)

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
        result_image = enhance_image(result.image, PostProcessingOptions(face_restorer_name="CodeFormer", restorer_visibility=1))
        
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
        with gr.Tab("Analyse Face"):
            img_to_analyse = gr.components.Image(type="pil", label="Face")
            analyse_det_threshold = gr.Slider(0.1, 1, 0.5, step=0.01, label="Detection threshold")
            analyse_btn = gr.Button("Analyse")
            analyse_results = gr.Textbox(label="Results", interactive=False, value="")

        upscale_options = upscaler_ui()

    explore_btn.click(explore_onnx_faceswap_model, inputs=[model], outputs=[explore_result_text])  
    compare_btn.click(compare, inputs=[img1, img2], outputs=[compare_result_text])
    generate_checkpoint_btn.click(build_face_checkpoint_and_save, inputs=[batch_files, name], outputs=[preview])
    extract_btn.click(extract_faces, inputs=[extracted_source_files, extract_save_path]+upscale_options, outputs=[extracted_faces])  
    analyse_btn.click(analyse_faces, inputs=[img_to_analyse,analyse_det_threshold], outputs=[analyse_results])  

def on_ui_tabs() :
    with gr.Blocks(analytics_enabled=False) as ui_faceswap:
        tools_ui()
    return [(ui_faceswap, "Roop", "roop_tab")] 

