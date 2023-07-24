from modules.face_restoration import FaceRestoration
from modules.upscaler import UpscalerData
from dataclasses import dataclass
from modules import shared
from enum import Enum

class InpaintingWhen(Enum):
    NEVER = "Never"
    BEFORE_UPSCALING = "Before Upscaling/all"
    BEFORE_RESTORE_FACE = "After Upscaling/Before Restore Face"
    AFTER_ALL = "After All"

@dataclass
class PostProcessingOptions:
    face_restorer_name: str = ""
    restorer_visibility: float = 0.5
    codeformer_weight: float = 1

    upscaler_name: str = ""
    scale: int = 1
    upscale_visibility: float = 0.5
    
    inpainting_denoising_strengh : float = 0
    inpainting_prompt : str = ""
    inpainting_negative_prompt : str = ""
    inpainting_steps : int = 20
    inpainting_sampler : str = "Euler"
    inpainting_when : InpaintingWhen = InpaintingWhen.BEFORE_UPSCALING
    inpainting_model : str = "Current"
    
    @property
    def upscaler(self) -> UpscalerData:
        for upscaler in shared.sd_upscalers:
            if upscaler.name == self.upscaler_name:
                return upscaler
        return None

    @property
    def face_restorer(self) -> FaceRestoration:
        for face_restorer in shared.face_restorers:
            if face_restorer.name() == self.face_restorer_name:
                return face_restorer
        return None