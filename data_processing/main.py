import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
import os

try:
    from dataset_reformatter import (
        clean_nan_from_dataset,
        create_dataset_with_caption_label_id,
        reformat_dataset,
    )
    from image_to_base64 import (
        create_dataset_with_base64_images,
        extract_image_id_from_image_path,
    )
    from processing_exceptions import MergeException

except ImportError:
    from .dataset_reformatter import (
        clean_nan_from_dataset,
        create_dataset_with_caption_label_id,
        reformat_dataset,
    )
    from .image_to_base64 import (
        create_dataset_with_base64_images,
        extract_image_id_from_image_path,
    )
    from .processing_exceptions import MergeException


def main():
    parser = argparse.ArgumentParser(description="A simple argparser with --save flag.")

    parser.add_argument("--save", action="store_true", help="Save data if flag is set.")
    parser.add_argument(
        "--path",
        type=str,
        default="../labels/project-10-at-2023-03-14-19-15-6c466213.csv",
        help="File path to the dataset in TSV format.",
    )

    args = parser.parse_args()

    shall_save: bool = args.save
    PATH: str = args.path

    dataset: pd.DataFrame = pd.read_csv(PATH, sep="\t")

    dataset_clean: pd.DataFrame = clean_nan_from_dataset(dataset)
    dataset_with_caption_label_id: pd.DataFrame = create_dataset_with_caption_label_id(
        dataset_clean
    )
    reformatted_dataset: pd.DataFrame = reformat_dataset(dataset_with_caption_label_id)

    dataset_with_images: pd.DataFrame = create_dataset_with_base64_images()
    dataset_with_images: pd.DataFrame = extract_image_id_from_image_path(
        dataset_with_images
    )

    merged_df = reformatted_dataset.merge(
        dataset_with_images[["image_id", "base64"]], on="image_id", how="left"
    )

    nan_encoded_image_rows = merged_df[merged_df["base64"].isna()]

    if len(nan_encoded_image_rows):
        raise MergeException("There is a merge conflict between the labels and images!")

    # Split the dataset into train (80%) and temp (20%)
    train_data, temp_data = train_test_split(merged_df, test_size=0.2, random_state=42)

    # Split the temp data into validation (10%) and test (10%)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    dataset_path = "../dataset/"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    if shall_save:
        FILE_NAME: str = f"{dataset_path}full_prepared_dataset.tsv"
        TRAIN_FILE_NAME: str = f"{dataset_path}train.tsv"
        TEST_FILE_NAME: str = f"{dataset_path}test.tsv"
        VAL_FILE_NAME: str = f"{dataset_path}val.tsv"

        print(f"Train data: {len(train_data)}")
        print(f"Test data: {len(test_data)}")
        print(f"Val data: {len(val_data)}")

        merged_df.to_csv(FILE_NAME, sep="\t", index=False)

        train_data.to_csv(TRAIN_FILE_NAME, sep="\t", index=False)
        val_data.to_csv(VAL_FILE_NAME, sep="\t", index=False)
        test_data.to_csv(TEST_FILE_NAME, sep="\t", index=False)


if __name__ == "__main__":
    main()
