import pandas as pd
from pyhere import here
import sys

sys.path.append(str(here().resolve()))
from dataset_reformatter import (
    _all_columns_are_present,
    create_dataset_with_caption_label_id,
    clean_nan_from_dataset,
    are_coords_and_texts_compliant,
    expand_rows,
    ColumnsNotPresentException,
    COLUMNS_FOR_FINE_TUNING,
)


def test_clean_nan_from_dataset():
    data = {"A": [1, 2, None], "B": [4, None, 6]}
    df = pd.DataFrame(data)

    cleaned_df = clean_nan_from_dataset(df)
    assert cleaned_df.shape[0] == 1
    assert cleaned_df.iloc[0]["A"] == 1


def test_are_coords_and_texts_compliant():
    data = {
        "image_id": [1, 2, 3],
        "prompt": [["a", "b"], ["c", "d"], ["e"]],
        "target_bounding_box": [["x1", "x2"], ["y1", "y2"], ["z"]],
    }
    df_compliant = pd.DataFrame(data)
    assert are_coords_and_texts_compliant(df_compliant)

    data["target_bounding_box"] = [["x1", "x2"], ["y1", "y2"], ["z", "z2"]]
    df_not_compliant = pd.DataFrame(data)
    assert not are_coords_and_texts_compliant(df_not_compliant)


def test_expand_rows():
    data = {
        "image_id": [1, 2],
        "prompt": [["a", "b"], ["c", "d"]],
        "target_bounding_box": [["x1", "x2"], ["y1", "y2"]],
    }
    df = pd.DataFrame(data)

    expanded_df = expand_rows(df)
    assert expanded_df.shape[0] == 4
    assert expanded_df["image_id"].nunique() == 2
