from scripts.roop_utils.models_utils import get_face_checkpoints
import gradio as gr

def faceswap_unit_ui(is_img2img, unit_num=1):
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
            
                face = gr.Dropdown(
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
                label="Swap in source image (blended face)",
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