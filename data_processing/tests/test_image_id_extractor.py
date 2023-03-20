from pyhere import here
import sys
import pandas as pd
import pytest

sys.path.append(str(here().resolve()))

from id_extractor import (
    extract_image_name,
    extract_image_id,
    extract_image_id_from_image_path,
)


@pytest.mark.parametrize(
    "input_image_name,expected_image_id",
    [
        ("example_image.png", "example_image"),
        ("12345.jpeg", "12345"),
        ("image_with-dash.jpg", "image_with"),
        ("image%name-Copy.jpg", "image"),
        ("image name-Copy.jpg", "image"),
    ],
)
def test_extract_image_id(input_image_name, expected_image_id):
    assert extract_image_id(input_image_name) == expected_image_id


def test_extract_image_name():
    assert extract_image_name("example.com/path/to/image.png") == "image.png"
    assert extract_image_name("/path/to/image2.jpg") == "image2.jpg"
    assert extract_image_name("image3.gif") == "image3.gif"
    assert extract_image_name("") == ""


def test_extract_image_id_from_image_path():
    data = [
        {"image": "/path/to/image_1234.jpg"},
        {"image": "/path/to/image_5678.png"},
        {"image": "/path/to/image_9012.jpeg"},
    ]
    input_df = pd.DataFrame(data)

    output_df = extract_image_id_from_image_path(input_df)

    assert "image_id" in output_df.columns
    assert "image_name" not in output_df.columns
    assert "image" not in output_df.columns
