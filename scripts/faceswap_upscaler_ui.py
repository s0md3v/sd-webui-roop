import gradio as gr
import modules
from modules import shared

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
                value=shared.face_restorers[0].name(),
                type="value",
            )
            with gr.Column():
                face_restorer_visibility = gr.Slider(
                    0, 1, 1, step=0.001, label="Restore visibility"
                )
                codeformer_weight = gr.Slider(
                    0, 1, 1, step=0.001, label="codeformer weight"
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
        codeformer_weight,
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