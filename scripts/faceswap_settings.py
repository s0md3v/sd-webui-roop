from scripts.roop_utils.models_utils import get_models
from modules import script_callbacks, shared
import gradio as gr

def on_ui_settings():
    section = ('roop', "Roop")
    models = get_models()
    shared.opts.add_option("roop_model", shared.OptionInfo(
        models[0] if len(models) > 0 else "None",  "Roop FaceSwap Model", gr.Dropdown, {"interactive": True, "choices" : models}, section=section)) 
    shared.opts.add_option("roop_keep_original", shared.OptionInfo(
        False, "keep original image before swapping", gr.Checkbox, {"interactive": True}, section=section))               
    shared.opts.add_option("roop_units_count", shared.OptionInfo(
        3, "Max faces units (requires restart)", gr.Slider, {"minimum": 1, "maximum": 10, "step": 1}, section=section))
    shared.opts.add_option("roop_upscaled_swapper", shared.OptionInfo(
        False, "Upscaled swapper", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("roop_upscaled_swapper_upscaler", shared.OptionInfo(
        None, "Upscaled swapper upscaler (Recommanded : LDSR)", gr.Dropdown, {"interactive": True, "choices" : [upscaler.name for upscaler in shared.sd_upscalers]}, section=section))
    shared.opts.add_option("roop_upscaled_swapper_sharpen", shared.OptionInfo(
        True, "Upscaled swapper sharpen", gr.Checkbox, {"interactive": True}, section=section))
    shared.opts.add_option("roop_upscaled_swapper_fixcolor", shared.OptionInfo(
        True, "Upscaled swapper color correction", gr.Checkbox, {"interactive": True}, section=section))    
    shared.opts.add_option("roop_upscaled_swapper_face_restorer", shared.OptionInfo(
        None, "Upscaled swapper face restorer", gr.Dropdown, {"interactive": True, "choices" : ["None"] + [x.name() for x in shared.face_restorers]}, section=section))
    shared.opts.add_option("roop_upscaled_swapper_face_restorer_visibility", shared.OptionInfo(
        1, "Upscaled swapper face restorer visibility", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.001}, section=section))
    shared.opts.add_option("roop_upscaled_swapper_face_restorer_weight", shared.OptionInfo(
        1, "Upscaled swapper face restorer weight (codeformer)", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.001}, section=section))

script_callbacks.on_ui_settings(on_ui_settings)