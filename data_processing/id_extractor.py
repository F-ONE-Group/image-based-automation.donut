import warnings
import pandas as pd
from typing import Tuple


def extract_image_name(image_path: str) -> str:
    image_name = image_path.split("/")[-1]
    return image_name


def extract_image_id(image_name: str) -> str:
    if "-" not in image_name:
        return image_name.split(".")[0]
    elif "Copy" in image_name:
        name_only: str = image_name.split("-")[0]
        if "%" in name_only:
            return name_only.split("%")[0]
        elif " " in name_only:
            return name_only.split(" ")[0]
    return image_name.split("-")[0]


def extract_image_id_from_image_path(dataset: pd.DataFrame) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset["image_name"] = dataset["image"].apply(extract_image_name)
        dataset["image_id"] = dataset["image_name"].apply(extract_image_id)

    redundant_columns: Tuple[str] = ("image_name", "image")
    for _redundant_column in redundant_columns:
        if _redundant_column in dataset.columns:
            dataset = dataset.drop(columns=[_redundant_column])
    return dataset
