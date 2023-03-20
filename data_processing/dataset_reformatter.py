import pandas as pd
from typing import Tuple
import warnings
import uuid

try:
    from id_extractor import (
        extract_image_id,
        extract_image_id_from_image_path,
    )
    from label_extractor import extract_labels_from_str, reformat_labels
    from text_extractor import reformat_text
    from processing_exceptions import (
        ColumnsNotPresentException,
        CoordsAndTextsNotCompliantException,
    )
except ImportError:
    from .id_extractor import (
        extract_image_id,
        extract_image_id_from_image_path,
    )
    from .label_extractor import extract_labels_from_str, reformat_labels
    from .text_extractor import reformat_text
    from .processing_exceptions import (
        ColumnsNotPresentException,
        CoordsAndTextsNotCompliantException,
    )


COLUMNS_FOR_FINE_TUNING: Tuple[str] = ("transcription", "label", "image")


def _all_columns_are_present(dataset: pd.DataFrame) -> bool:
    return any(
        _column not in COLUMNS_FOR_FINE_TUNING for _column in dataset.columns.to_list()
    )


def create_dataset_with_caption_label_id(dataset: pd.DataFrame) -> pd.DataFrame:
    if not _all_columns_are_present(dataset):
        raise ColumnsNotPresentException(
            "any of the caption, image or id columns are not present"
        )

    dataset_with_caption_label_id = dataset[list(COLUMNS_FOR_FINE_TUNING)]
    return dataset_with_caption_label_id


def clean_nan_from_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    ### finds NaNs and rm them
    rows_with_nan = dataset[dataset.isna().any(axis=1)].index
    if rows_with_nan is not None:
        print(f"found {len(rows_with_nan)} NaNs")
        print("these rows contain NaNs: ")
        print(rows_with_nan)
        return dataset.dropna()
    return dataset


def are_coords_and_texts_compliant(df: pd.DataFrame) -> bool:
    for _, row in df.iterrows():
        if isinstance(row["prompt"], str):
            text_length = 1
        else:
            text_length = len(row["prompt"])
        region_coord_length = len(row["target_bounding_box"])

        if text_length != region_coord_length:
            print("these fields are not compliant: ")
            print("prompt: ", row["prompt"])
            print("target_bounding_box: ", row["target_bounding_box"])
            return False
    return True


# Function to expand rows
def expand_rows(dataset: pd.DataFrame) -> pd.DataFrame:
    expanded_data = []

    for _, row in dataset.iterrows():
        image_id = row["image_id"]

        for prompt, coord in zip(row["prompt"], row["target_bounding_box"]):
            expanded_data.append(
                {
                    "image_id": image_id,
                    # "uniq-id": uniq_id,
                    "prompt": prompt,
                    "target_bounding_box": coord,
                }
            )

    expanded_df = pd.DataFrame(expanded_data)
    return expanded_df


def reformat_dataset(dataset_with_caption_label_id: pd.DataFrame) -> pd.DataFrame:
    print("the columns: ")
    print(dataset_with_caption_label_id.columns.to_list())

    dataset_with_caption_label_id = extract_image_id_from_image_path(
        dataset_with_caption_label_id
    )

    ### rename id to uniq-id
    # dataset_with_caption_label_id["uniq-id"] = dataset_with_caption_label_id["id"]
    # if "id" in dataset_with_caption_label_id.columns:
    #     dataset_with_caption_label_id = dataset_with_caption_label_id.drop(
    #         columns=["id"]
    #     )

    ### rename transcription to text
    dataset_with_caption_label_id["prompt"] = dataset_with_caption_label_id[
        "transcription"
    ]
    if "transcription" in dataset_with_caption_label_id.columns:
        dataset_with_caption_label_id = dataset_with_caption_label_id.drop(
            columns=["transcription"]
        )

    ### reformat labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset_with_caption_label_id["label_dict"] = dataset_with_caption_label_id[
            "label"
        ].apply(extract_labels_from_str)
        dataset_with_caption_label_id["target_bounding_box"] = dataset_with_caption_label_id[
            "label_dict"
        ].apply(reformat_labels)

    redundant_label_columns: Tuple[str] = ("label_dict", "label")
    for _redundant_label_column in redundant_label_columns:
        if _redundant_label_column in dataset_with_caption_label_id.columns:
            dataset_with_caption_label_id = dataset_with_caption_label_id.drop(
                columns=[_redundant_label_column]
            )

    ### reformat text
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset_with_caption_label_id["prompt"] = dataset_with_caption_label_id[
            "prompt"
        ].apply(reformat_text)

    if not are_coords_and_texts_compliant(dataset_with_caption_label_id):
        raise CoordsAndTextsNotCompliantException("Coords and prompt are not compliant!")

    expanded_df = expand_rows(dataset_with_caption_label_id)

    return expanded_df
