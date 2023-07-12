from scripts.roop_logging import logger
import os

VERSION_FLAG = "v0.1.0"
EXTENSION_PATH=os.path.join("extensions","sd-webui-roop")
SD_CONVERT_SCORE = 0.7

logger.info(f"Roop {VERSION_FLAG}")
