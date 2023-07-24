## 1.1.0 :
All listed in features

+ add inpainting model selection => allow to select a different model for face inpainting
+ add source faces selection => allow to select the reference face if multiple face are present in reference image
+ add select by size => sort faces by size from larger to smaller
+ add batch option => allow to process images without txt2img or i2i in tabs
+ add segmentation mask for upscaled inpainter (based on codeformer implementation) : avoid square mask and prevent degradation of non-face parts of the image.

## 0.1.0 :

### Major :
+ add multiple face support
+ add face blending support (will blend sources faces)
+ add face similarity evaluation (will compare face to a reference)
    + add filters to discard images that are not rated similar enough to reference image and source images
+ add face tools tab
    + face extraction tool
    + face builder tool : will build a face model that can be reused
+ add faces models

### Minor :

Improve performance by not reprocessing source face each time

### Breaking changes

base64 and api not supported anymore (will be reintroduced in the future)