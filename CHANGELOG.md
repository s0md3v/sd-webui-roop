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