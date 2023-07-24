"""
Code from codeformer https://github.com/sczhou/CodeFormer
"""

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import normalize
from scripts.roop_swapping.parsing import init_parsing_model
from functools import lru_cache

@lru_cache
def get_parsing_model(device) :
    return init_parsing_model(device=device)

def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)

def generate_face_mask(face_img, device):    
    # Redimensionner l'image du visage pour le modèle
    face_input = cv2.resize(face_img, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    # Prétraitement de l'image
    face_input = img2tensor(face_input.astype('float32') / 255., bgr2rgb=True, float32=True)
    normalize(face_input, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    face_input = torch.unsqueeze(face_input, 0).to(device)

    # Faire passer l'image à travers le modèle
    with torch.no_grad():
        out = get_parsing_model(device)(face_input)[0]
    out = out.argmax(dim=1).squeeze().cpu().numpy()

    # Générer le masque à partir de la sortie du modèle
    parse_mask = np.zeros(out.shape)
    MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 255, 0, 0, 0]
    for idx, color in enumerate(MASK_COLORMAP):
        parse_mask[out == idx] = color

    # Redimensionner le masque pour qu'il corresponde à l'image d'origine
    face_mask = cv2.resize(parse_mask, (face_img.shape[1], face_img.shape[0]))

    return face_mask