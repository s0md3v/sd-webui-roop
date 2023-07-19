
import glob
import os
import modules.scripts as scripts
from modules import scripts
from scripts.roop_globals import EXTENSION_PATH
from modules.shared import opts
from scripts.roop_logging import logger

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

def get_current_model() -> str :
    model = opts.data.get("roop_model", None)
    if model is None :
        models = get_models()
        model = models[0] if len(models) else None
    logger.info("Try to use model : %s", model)
    if not os.path.isfile(model):
        logger.error("The model %s cannot be found or loaded", model)
        raise FileNotFoundError("No faceswap model found. Please add it to the roop directory.")
    return model

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