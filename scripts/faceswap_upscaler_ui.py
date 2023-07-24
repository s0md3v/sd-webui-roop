import gradio as gr
import modules
from modules import shared, sd_models
from modules.shared import cmd_opts, opts, state

import scripts.roop_postprocessing.upscaling as upscaling
from scripts.roop_logging import logger

def upscaler_ui():
    with gr.Tab(f"Post-Processing"):
        gr.Markdown(
                """Upscaling is performed on the whole image. Upscaling happens before face restoration.""")
        with gr.Row():
            face_restorer_name = gr.Radio(
                label="Restore Face",
                choices=["None"] + [x.name() for x in shared.face_restorers],
                value=lambda : opts.data.get("roop_pp_default_face_restorer", shared.face_restorers[0].name()),
                type="value",
                elem_id="roop_pp_face_restorer"
            )
            with gr.Column():
                face_restorer_visibility = gr.Slider(
                    0, 1, value=lambda:opts.data.get("roop_pp_default_face_restorer_visibility", 1), step=0.001, label="Restore visibility",
                    elem_id="roop_pp_face_restorer_visibility"
                )
                codeformer_weight = gr.Slider(
                    0, 1, value=lambda:opts.data.get("roop_pp_default_face_restorer_weight", 1), step=0.001, label="codeformer weight",
                    elem_id="roop_pp_face_restorer_weight"
                )                
        upscaler_name = gr.Dropdown(
            choices=[upscaler.name for upscaler in shared.sd_upscalers],
            value= lambda:opts.data.get("roop_pp_default_upscaler","None"),
            label="Upscaler",
            elem_id="roop_pp_upscaler"
        )
        upscaler_scale = gr.Slider(1, 8, 1, step=0.1, label="Upscaler scale", elem_id="roop_pp_upscaler_scale")
        upscaler_visibility = gr.Slider(
            0, 1, value=lambda:opts.data.get("roop_pp_default_upscaler_visibility", 1), step=0.1, label="Upscaler visibility (if scale = 1)",
            elem_id="roop_pp_upscaler_visibility"
        )
        with gr.Accordion(f"Post Inpainting", open=True):
            gr.Markdown(
                """Inpainting sends image to inpainting with a mask on face (once for each faces).""")
            inpainting_when = gr.Dropdown(
            elem_id="roop_pp_inpainting_when", choices = [e.value for e in upscaling.InpaintingWhen.__members__.values()],value=[upscaling.InpaintingWhen.BEFORE_RESTORE_FACE.value], label="Enable/When")
            inpainting_denoising_strength = gr.Slider(
                0, 1, 0, step=0.01, elem_id="roop_pp_inpainting_denoising_strength", label="Denoising strenght (will send face to img2img after processing)"
            )

            inpainting_denoising_prompt = gr.Textbox("Portrait of a [gender]",elem_id="roop_pp_inpainting_denoising_prompt", label="Inpainting prompt use [gender] instead of men or woman")
            inpainting_denoising_negative_prompt = gr.Textbox("", elem_id="roop_pp_inpainting_denoising_neg_prompt", label="Inpainting negative prompt use [gender] instead of men or woman")
            with gr.Row():
                samplers_names = [s.name for s in modules.sd_samplers.all_samplers]
                inpainting_sampler = gr.Dropdown(
                        choices=samplers_names,
                        value=[samplers_names[0]],
                        label="Inpainting Sampler",
                        elem_id="roop_pp_inpainting_sampler"
                    )
                inpainting_denoising_steps =  gr.Slider(
                    1, 150, 20, step=1, label="Inpainting steps",
                    elem_id="roop_pp_inpainting_steps"
                )
                
            inpaiting_model = gr.Dropdown(choices=["Current"]+sd_models.checkpoint_tiles(), default="Current", label="sd model (experimental)", elem_id="roop_pp_inpainting_sd_model")
    return [
        face_restorer_name,
        face_restorer_visibility,
        codeformer_weight,
        upscaler_name,
        upscaler_scale,
        upscaler_visibility,
        inpainting_denoising_strength,
        inpainting_denoising_prompt,
        inpainting_denoising_negative_prompt,
        inpainting_denoising_steps,
        inpainting_sampler,
        inpainting_when,
        inpaiting_model
    ]