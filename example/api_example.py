import base64
import io
import requests

address = 'http://127.0.0.1:7860'
image_file = "path/to/local/image/file"
im = Image.open(image_file)

img_bytes = io.BytesIO()
im.save(img_bytes, format='PNG') 
img_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')

# the most important is this one
args=[img_base64,True,'0','F:\stable-diffusion-webui\models/roop\inswapper_128.onnx','CodeFormer',1,None,1,'None',False,True]

# The args for roop can be found by 
# requests.get(url=f'{address}/sdapi/v1/script-info')

prompt = "(8k, best quality, masterpiece, ultra highres:1.2),Realistic image style,Vertical orientation, Girl,White short hair,Shoulder-length, tousled White blossom pink hair,Clothing"
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
    #"script_name":"roop", # this doesn't work
    #"script_args":args, # this doesn't work
    "alwayson_scripts": {"roop":{"args":args}}  # This is the args.
}
result = requests.post(url=f'{address}/sdapi/v1/txt2img', json=payload)