import base64
import os
from io import BytesIO
from PIL import Image

from pyhere import here
import sys

sys.path.append(str(here().resolve()))

from image_to_base64 import image_to_base64
from typing import Final

MSE_THRESHOLD: Final[int] = 1


def test_image_to_base64():
    img = Image.new("RGB", (10, 10), color="red")
    img_path = "test_image.jpg"
    img.save(img_path)

    try:
        base64_str = image_to_base64(img_path)

        byte_data = base64.b64decode(base64_str)
        img_buffer = BytesIO(byte_data)
        decoded_img = Image.open(img_buffer)

        assert img.size == decoded_img.size
        assert img.mode == decoded_img.mode

        mse = 0
        for x in range(img.width):
            for y in range(img.height):
                orig_pixel = img.getpixel((x, y))
                decoded_pixel = decoded_img.getpixel((x, y))
                mse += sum(
                    (orig - decoded) ** 2
                    for orig, decoded in zip(orig_pixel, decoded_pixel)
                )
        mse /= img.width * img.height * len(img.mode)

        assert mse < MSE_THRESHOLD  # Adjustable param

    finally:
        if os.path.exists(img_path):
            os.remove(img_path)
