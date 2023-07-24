"""
Code from codeformer https://github.com/sczhou/CodeFormer
"""


import torch
import cv2
import os
import os.path as osp
import torch
from torch.hub import download_url_to_file, get_dir
from .parsenet import ParseNet
from urllib.parse import urlparse
from scripts.roop_globals import FACE_PARSER_DIR

ROOT_DIR = FACE_PARSER_DIR

def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
    """
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(os.path.join(ROOT_DIR, model_dir), exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(ROOT_DIR, model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def init_parsing_model(device='cuda'):
    model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
    model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth'
    model_path = load_file_from_url(url=model_url, model_dir='weights/facelib', progress=True, file_name=None)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
