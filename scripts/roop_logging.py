import logging
import copy
import sys
from modules import shared
from PIL import Image

class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[0;36m",  # CYAN
        "INFO": "\033[0;32m",  # GREEN
        "WARNING": "\033[0;33m",  # YELLOW
        "ERROR": "\033[0;31m",  # RED
        "CRITICAL": "\033[0;37;41m",  # WHITE ON RED
        "RESET": "\033[0m",  # RESET COLOR
    }

    def format(self, record):
        colored_record = copy.copy(record)
        levelname = colored_record.levelname
        seq = self.COLORS.get(levelname, self.COLORS["RESET"])
        colored_record.levelname = f"{seq}{levelname}{self.COLORS['RESET']}"
        return super().format(colored_record)


# Create a new logger
logger = logging.getLogger("Roop")
logger.propagate = False

# Add handler if we don't have one.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)

# Configure logger
loglevel_string = getattr(shared.cmd_opts, "roop_loglevel", "INFO")
loglevel = getattr(logging, loglevel_string.upper(), "INFO")
logger.setLevel(loglevel)

import tempfile
if logger.getEffectiveLevel() <= logging.DEBUG :
    DEBUG_DIR = tempfile.mkdtemp()

def save_img_debug(img: Image.Image, message: str, *opts):
    if logger.getEffectiveLevel() <= logging.DEBUG:
        with tempfile.NamedTemporaryFile(dir=DEBUG_DIR, delete=False, suffix=".png") as temp_file:
            img_path = temp_file.name
            img.save(img_path)

        message_with_link = f"{message}\nImage: file://{img_path}"
        logger.debug(message_with_link, *opts)