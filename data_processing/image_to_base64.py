from PIL import Image
from io import BytesIO
import base64
import pandas as pd
import os
from pyhere import here
import sys

sys.path.append(str(here().resolve()))

try:
    from id_extractor import (
        extract_image_id,
        extract_image_name,
        extract_image_id_from_image_path,
    )
except ImportError:
    from .id_extractor import (
        extract_image_id,
        extract_image_name,
        extract_image_id_from_image_path,
    )


IMAGES_PATH: str = "../images/"
IMAGE_FORMAT: str = ".png"


def image_to_base64(file_path: str) -> str:
    img = Image.open(file_path)  # path to file
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)  # bytes
    base64_str = base64_str.decode("utf-8")  # str
    return base64_str


def create_dataset_with_base64_images(
    images_directory: str = IMAGES_PATH,
) -> pd.DataFrame:
    data = []

    for file_name in os.listdir(images_directory):
        file_path = os.path.join(images_directory, file_name)

        if os.path.isfile(file_path) and file_name.lower().endswith((IMAGE_FORMAT)):
            base64_str = image_to_base64(file_path)
            data.append({"image": file_path, "base64": base64_str})

    dataset = pd.DataFrame(data)
    return dataset


if __name__ == "__main__":
    dataset: pd.DataFrame = create_dataset_with_base64_images(IMAGES_PATH)
    dataset: pd.DataFrame = extract_image_id_from_image_path(dataset)
