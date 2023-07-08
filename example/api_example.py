import base64
import io
import requests
from PIL import Image
import os
import base64, io


address = 'http://127.0.0.1:7860'
image_file = "../references/man.png"
im = Image.open(image_file)

img_bytes = io.BytesIO()
im.save(img_bytes, format='PNG') 
img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

models_dir = os.path.abspath(os.path.join("models","roop"))
# the most important is this one
args=[
    img_base64, # Reference
    None, # Face Checkpoint
    None, # Batch
    False, # Blend
    True, # Enable
    True, # Same Gender
    0, # Min sim
    0, # Min ref sim
    '0', # Face Number
    False, # Swap in source
    True # Swap in generated
] 

# The args for roop can be found by 
# requests.get(url=f'{address}/sdapi/v1/script-info')

prompt = "(8k, best quality, masterpiece, ultra highres:1.2),Realistic image style,Vertical orientation, Man wearing suit, Einstein"
neg = "ng_deepnegative_v1_75t, (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, normal quality, ((monochrome)), ((grayscale)), (verybadimagenegative_v1.3:0.8), negative_hand-neg, (lamp), badhandv4"
payload = {
    "prompt": prompt,
    "negative_prompt": neg,
    "seed": -1,
    "sampler_name": "DPM++ SDE Karras",
    "steps": 20,
    "cfg_scale": 7,
    "width": 512,
    "height": 768,
    "restore_faces": True,
    "alwayson_scripts": {"roop":{"args":args}}
}  

result = requests.post(url=f'{address}/sdapi/v1/txt2img', json=payload)
images = result.json()["images"]
print(len(images), "images generated")
for i in images :
    img_bytes = base64.b64decode(i)
    img = Image.open(io.BytesIO(img_bytes))
    img.show()
