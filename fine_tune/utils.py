#@title Let's look at a sample in the dataset
import math
import json
from PIL import Image, ImageDraw, ImageOps

import base64
from io import BytesIO
import numpy as np
from typing import Tuple

def base64_to_pil_image(base64_string):
    # Decode the base64 string to bytes
    image_bytes = base64.b64decode(base64_string)

    # Open a BytesIO stream to read the decoded bytes
    stream = BytesIO(image_bytes)

    # Open the image using PIL and return it
    return Image.open(stream)


def resize_and_pad_image(img, size: Tuple[int, int]):
    """
    Resize an image to a target size and add padding if necessary.

    Args:
        img (PIL.Image): The input image.
        size (tuple): The target size, as a (width, height) tuple.

    Returns:
        PIL.Image: The resized and padded image.
    """
    # Compute the aspect ratio of the original image.
    aspect_ratio = img.size[0] / img.size[1]

    # Compute the aspect ratio of the target size.
    target_aspect_ratio = size[0] / size[1]

    # If the aspect ratios are the same, just resize the image.
    if np.isclose(aspect_ratio, target_aspect_ratio):
        return img.resize(size, Image.BICUBIC)

    # Compute the size of the image after resizing to fit the width.
    new_size = (int(size[0]), int(size[0] / aspect_ratio))

    # If the resulting height is less than the target height, add padding to the top and bottom.
    if new_size[1] < size[1]:
        padding_top = (size[1] - new_size[1]) // 2
        padding_bottom = size[1] - new_size[1] - padding_top
        img = img.resize(new_size, Image.BICUBIC)
        return ImageOps.expand(img, (0, padding_top, 0, padding_bottom), fill=0)

    # Otherwise, add padding to the left and right.
    padding_left = (new_size[0] - size[0]) // 2
    padding_right = new_size[0] - size[0] - padding_left
    img = img.resize(new_size, Image.BICUBIC)
    return ImageOps.expand(img, (padding_left, 0, padding_right, 0), fill=0)
