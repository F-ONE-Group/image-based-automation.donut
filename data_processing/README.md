# Data Preparation Module
This script prepares the RefCoco dataset for use in a machine learning model. It reads in a dataset in TSV format from a specified file path, cleans it, and reformats it to include the necessary caption, label, and image data. The script then splits the dataset into training, validation, and test sets, and saves the resulting datasets to separate TSV files.

# Usage
To reproduce the results, please follow these steps:

1. Create a new conda environment: ```conda create --name refcoco_env```.
2. Activate the environment: ```conda activate refcoco_env```.
3. Install the required dependencies from the requirements.txt file: ```conda install --file requirements.txt```.
4. Run the script: ```python script.py --save --path <path_to_the_labeled_dataset/dataset.tsv```.

The ```--save``` flag is optional and can be used to save the resulting datasets to separate TSV files. The ```--path``` flag is also optional and can be used to specify the file path to the dataset in TSV format. If not specified, the default value ```"../coco_data/project-10-at-2023-03-14-19-15-6c466213.csv"``` will be used which will fail on your local machine since this file is not included in git.
You can specify a different file path by passing the --path argument followed by the file path.

# Dependencies
This script requires the following dependencies:

- argparse: For parsing command-line arguments.
- pandas: For reading and manipulating tabular data.
- sklearn: For splitting the dataset into training, validation, and test sets.

# Output
The script outputs the following files:

full_prepared_dataset.tsv: A TSV file containing the full prepared dataset.
- refcoco_train.tsv: A TSV file containing the training set.
- refcoco_val.tsv: A TSV file containing the validation set.
- refcoco_test.tsv: A TSV file containing the test set.