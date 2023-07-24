from scripts.roop_logging import logger
import os

MODELS_DIR = os.path.abspath(os.path.join("models","roop"))
ANALYZER_DIR = os.path.abspath(os.path.join(MODELS_DIR, "analysers"))
FACE_PARSER_DIR = os.path.abspath(os.path.join(MODELS_DIR, "parser"))

VERSION_FLAG = "v1.1.0"
EXTENSION_PATH=os.path.join("extensions","sd-webui-roop")
SD_CONVERT_SCORE = 0.7

