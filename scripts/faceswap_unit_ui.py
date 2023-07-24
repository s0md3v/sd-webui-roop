from scripts.roop_utils.models_utils import get_face_checkpoints
import gradio as gr

def faceswap_unit_ui(is_img2img, unit_num=1, id_prefix="roop"):
    with gr.Tab(f"Face {unit_num}"):
        with gr.Column():
            gr.Markdown(
            """Reference is an image. First face will be extracted. 
            First face of batches sources will be extracted and used as input (or blended if blend is activated).""")
            with gr.Row():
                img = gr.components.Image(type="pil", label="Reference", elem_id=f"{id_prefix}_face{unit_num}_reference_image")
                batch_files = gr.components.File(
                    type="file",
                    file_count="multiple",
                    label="Batch Sources Images",
                    optional=True,
                    elem_id=f"{id_prefix}_face{unit_num}_batch_source_face_files"
                )
            gr.Markdown(
                """Face checkpoint built with the checkpoint builder in tools. Will overwrite reference image.""")     
            with gr.Row() :
            
                face = gr.Dropdown(
                    choices=get_face_checkpoints(),
                    label="Face Checkpoint (precedence over reference face)",
                    elem_id=f"{id_prefix}_face{unit_num}_face_checkpoint"
                )
                refresh = gr.Button(value='↻', variant='tool', elem_id=f"{id_prefix}_face{unit_num}_refresh_checkpoints")
                def refresh_fn(selected):
                    return gr.Dropdown.update(value=selected, choices=get_face_checkpoints())
                refresh.click(fn=refresh_fn,inputs=face, outputs=face)

            with gr.Row():
                enable = gr.Checkbox(False, placeholder="enable", label="Enable", elem_id=f"{id_prefix}_face{unit_num}_enable")
                blend_faces = gr.Checkbox(
                    True, placeholder="Blend Faces", label="Blend Faces ((Source|Checkpoint)+References = 1)",
                    elem_id=f"{id_prefix}_face{unit_num}_blend_faces",
                    interactive=True
                )
            gr.Markdown("""Discard images with low similarity or no faces :""")
            with gr.Row():
                check_similarity = gr.Checkbox(False, placeholder="discard", label="Check similarity",
                    elem_id=f"{id_prefix}_face{unit_num}_check_similarity")  
                compute_similarity = gr.Checkbox(False, label="Compute similarity",
                    elem_id=f"{id_prefix}_face{unit_num}_compute_similarity")      
            min_sim = gr.Slider(0, 1, 0, step=0.01, label="Min similarity",
                    elem_id=f"{id_prefix}_face{unit_num}_min_similarity")
            min_ref_sim = gr.Slider(
                0, 1, 0, step=0.01, label="Min reference similarity",
                    elem_id=f"{id_prefix}_face{unit_num}_min_ref_similarity"
            )

            gr.Markdown("""Select the face to be swapped, you can sort by size or use the same gender as the desired face:""")
            with gr.Row():
                same_gender = gr.Checkbox(
                    False, placeholder="Same Gender", label="Same Gender",
                    elem_id=f"{id_prefix}_face{unit_num}_same_gender"
                )
                sort_by_size = gr.Checkbox(
                    False, placeholder="Sort by size", label="Sort by size (larger>smaller)",
                    elem_id=f"{id_prefix}_face{unit_num}_sort_by_size"
                )
            target_faces_index = gr.Textbox(
                value="0",
                placeholder="Which face to swap (comma separated), start from 0 (by gender if same_gender is enabled)",
                label="Target face : Comma separated face number(s)",
                elem_id=f"{id_prefix}_face{unit_num}_target_faces_index"
            )
            gr.Markdown("""The following will only affect reference face image (and is not affected by sort by size) :""")
            reference_faces_index = gr.Number(
                value=0,
                precision=0,
                minimum=0,
                placeholder="Which face to get from reference image start from 0",
                label="Reference source face : start from 0",
                elem_id=f"{id_prefix}_face{unit_num}_reference_face_index"
            )
            gr.Markdown("""Configure swapping. Swapping can occure before img2img, after or both :""", visible=is_img2img)        
            swap_in_source = gr.Checkbox(
                False,
                placeholder="Swap face in source image",
                label="Swap in source image (blended face)",
                visible=is_img2img,
                elem_id=f"{id_prefix}_face{unit_num}_swap_in_source"
            )
            swap_in_generated = gr.Checkbox(
                True,
                placeholder="Swap face in generated image",
                label="Swap in generated image",
                visible=is_img2img,
                elem_id=f"{id_prefix}_face{unit_num}_swap_in_generated"
            )
    # If changed, you need to change FaceSwapUnitSettings accordingly
    # ORDER of parameters is IMPORTANT. It should match the result of FaceSwapUnitSettings
    return [
        img,
        face,
        batch_files,
        blend_faces,
        enable,
        same_gender,
        sort_by_size,
        check_similarity,
        compute_similarity,
        min_sim,
        min_ref_sim,
        target_faces_index,
        reference_faces_index,
        swap_in_source,
        swap_in_generated,
    ]