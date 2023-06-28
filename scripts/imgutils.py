from PIL import Image
import cv2
import numpy as np
from math import isqrt, ceil

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_img):
    return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))



def torch_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def pil_to_torch(pil_images):
    """
    Convert a PIL image or a list of PIL images to a torch tensor or a batch of torch tensors.
    """
    if isinstance(pil_images, list):
        numpy_images = [np.array(image) for image in pil_images]
        torch_images = torch.from_numpy(np.stack(numpy_images)).permute(0, 3, 1, 2)
        return torch_images

    numpy_image = np.array(pil_images)
    torch_image = torch.from_numpy(numpy_image).permute(2, 0, 1)
    return torch_image



def create_square_image(image_list):
    size = None
    for image in image_list:
        if size is None:
            size = image.size
        elif image.size != size:
            raise ValueError("Not same size images")

    num_images = len(image_list)
    rows = isqrt(num_images)
    cols = ceil(num_images / rows)

    square_size = (cols * size[0], rows * size[1])

    square_image = Image.new("RGB", square_size)

    for i, image in enumerate(image_list):
        row = i // cols
        col = i % cols

        square_image.paste(image, (col * size[0], row * size[1]))

    return square_image
