import base64
import io
import requests
from PIL import Image
from client_utils import FaceSwapRequest, FaceSwapUnit, PostProcessingOptions, FaceSwapResponse, pil_to_base64

address = 'http://127.0.0.1:7860'

# First face unit :
unit1 = FaceSwapUnit(
    source_img=pil_to_base64("../../references/man.png"), # The face you want to use
    faces_index=(0,) # Replace first face
)

# Second face unit :
unit2 = FaceSwapUnit(
    source_img=pil_to_base64("../../references/woman.png"), # The face you want to use
    same_gender=True,
    faces_index=(0,) # Replace first woman since same gender is on
)

# Post-processing config :
pp = PostProcessingOptions(
    face_restorer_name="CodeFormer",
    codeformer_weight=0.5,
    restorer_visibility= 1)

# Prepare the request 
request = FaceSwapRequest (
    image = pil_to_base64("test_image.png"),
    units= [unit1, unit2], 
    postprocessing=pp
)


result = requests.post(url=f'{address}/roop/swap_face', data=request.json(), headers={"Content-Type": "application/json; charset=utf-8"})
response = FaceSwapResponse.parse_obj(result.json())

for img, info in zip(response.pil_images, response.infos):
    img.show(title = info)

