from donut_dataset import DonutDataset
from donut_lightning_module import DonutModelPLModule
# from prepare_dataset import desktop_dataset_dict
from torch.utils.data import DataLoader
import os
from pytorch_lightning.callbacks import StochasticWeightAveraging, LearningRateMonitor
from transformers import DonutProcessor, VisionEncoderDecoderModel, BartConfig, VisionEncoderDecoderConfig
from pytorch_lightning.callbacks import ProgressBar
import json

DATA_PATH: str = "../desktop_dataset_dict.json"

from typing import List


import pandas as pd
import ast
from utils import base64_to_pil_image, resize_and_pad_image
from typing import Tuple
import json

PATH: str = "../dataset/full_prepared_dataset.tsv"
PATH_TRAIN: str = "../dataset/train.tsv"
PATH_TEST: str = "../dataset/test.tsv"
PATH_VAL: str = "../dataset/val.tsv"
SEP: str = "\t"
desktop_dataset = pd.read_csv(PATH, sep=SEP)
train_dataset = pd.read_csv(PATH_TRAIN, sep=SEP)
test_dataset = pd.read_csv(PATH_TEST, sep=SEP)
val_dataset = pd.read_csv(PATH_VAL, sep=SEP)


desktop_dataset["target_bounding_box"] = desktop_dataset["target_bounding_box"].apply(ast.literal_eval)
train_dataset["target_bounding_box"] = train_dataset["target_bounding_box"].apply(ast.literal_eval)
test_dataset["target_bounding_box"] = test_dataset["target_bounding_box"].apply(ast.literal_eval)
val_dataset["target_bounding_box"] = val_dataset["target_bounding_box"].apply(ast.literal_eval)


desktop_dataset["image"] = desktop_dataset["base64"].apply(base64_to_pil_image)
desktop_dataset = desktop_dataset.drop(columns=["base64"])

train_dataset["image"] = train_dataset["base64"].apply(base64_to_pil_image)
train_dataset = train_dataset.drop(columns=["base64"])

test_dataset["image"] = test_dataset["base64"].apply(base64_to_pil_image)
test_dataset = test_dataset.drop(columns=["base64"])

val_dataset["image"] = val_dataset["base64"].apply(base64_to_pil_image)
val_dataset = val_dataset.drop(columns=["base64"])


# Apply the resize_and_pad_image function to the 'image' column.
SIZE: Tuple[int, int] = (1920, 1080)

desktop_dataset["image"] = desktop_dataset["image"].apply(lambda x: resize_and_pad_image(x, SIZE))

train_dataset["image"] = train_dataset["image"].apply(lambda x: resize_and_pad_image(x, SIZE))

test_dataset["image"] = test_dataset["image"].apply(lambda x: resize_and_pad_image(x, SIZE))

val_dataset["image"] = val_dataset["image"].apply(lambda x: resize_and_pad_image(x, SIZE))

values = {}
for i in range(0, len(desktop_dataset) - 1):
    values[i] = desktop_dataset.iloc[i].to_dict()

train_values = {}
for i in range(0, len(train_dataset) - 1):
    train_values[i] = train_dataset.iloc[i].to_dict()

test_values = {}
for i in range(0, len(test_dataset) - 1):
    test_values[i] = test_dataset.iloc[i].to_dict()

val_values = {}
for i in range(0, len(val_dataset) - 1):
    val_values[i] = val_dataset.iloc[i].to_dict()


desktop_dataset_dict = {"train": train_values, "test": test_values, "validation": val_values}


if __name__ == "__main__":
    print("Started execution")

    with open(DATA_PATH, "r") as infile:
        desktop_dataset_dict = json.load(infile)

    print("Loaded data")

    REFEXP_DATASET_NAME = "ivelin/rico_refexp_combined"

    # Pick which pretrained checkpoint to start the fine tuning process from
    REFEXP_MODEL_CHECKPOINT = 'ivelin/donut-refexp-combined-v1'
    REFEXP_MODEL_CP_BRANCH = 'main'


    pretrained_repo_name = REFEXP_MODEL_CHECKPOINT
    pretrained_repo_branch = REFEXP_MODEL_CP_BRANCH

    max_length = 128
    # image_size = [1280, 960]
    image_size = [1920, 1080]

    # update image_size of the encoder
    # during pre-training, a larger image size was used
    config = VisionEncoderDecoderConfig.from_pretrained(pretrained_repo_name, branch=pretrained_repo_branch)
    config.encoder.image_size = image_size  # (height, width)
    # update max_length of the decoder (for generation)
    config.decoder.max_length = max_length
    processor = DonutProcessor.from_pretrained(pretrained_repo_name, revision=pretrained_repo_branch)
    model = VisionEncoderDecoderModel.from_pretrained(pretrained_repo_name, revision=pretrained_repo_branch, config=config)

    train_batch_size: int = 1 # Usually increments of 8. Value depends on GPU capacity.
    print(f"train_batch_size: {train_batch_size}")
    val_batch_size: int = 1
    # Since the whole dataset is too big to train in a single epoch
    # We will sample a small subset (5%-10%) per loop and train for a few epochs
    # Then sample again and loop a few more epochs
    # In effect simulating training on the whole dataset.
    max_epochs_per_loop=10 # previously at 30 epochs and 1024 training samples
    print(f"max_epochs_per_loop: {max_epochs_per_loop}")

    num_training_samples_per_epoch=800 # initially 800
    print(f"num_training_samples_per_epoch: {num_training_samples_per_epoch}")

    # Start at 3e-5 and reduce gradually every few epochs if loss oscilations too high. Use LR scheduler if epochs > 10.
    # See scheduler docs: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html
    learning_rate= 3e-6 # previously , 1e-6, 1e-5
    print(f"learning_rate: {learning_rate}")

    # Aim for 10%. Examples: 20 = 800/8*2/10, 10%; 300 for 800/8*30/10
    warmup_steps=(num_training_samples_per_epoch/train_batch_size)*max_epochs_per_loop/10
    print(f"warmup_steps: {warmup_steps}")


    experiment_name = "DonutUI"
    save_dir = "./donutUI/v1/"
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # we update some settings which differ from pretraining; namely the size of the images + no rotation required
    # source: https://github.com/clovaai/donut/blob/master/config/train_cord.yaml
    processor.feature_extractor.size = image_size[::-1] # should be (width, height)
    processor.feature_extractor.do_align_long_axis = False

    # For warm up phase, consider picking only a small subset to see if the model converges on the data
    train_dataset = DonutDataset(desktop_dataset_dict,
                                 max_length=max_length,
                                 split="train",
                                 task_start_token="<s_refexp>",
                                 prompt_end_token="<s_target_bounding_box>",
                                 sort_json_key=False,
                                 )

    val_dataset = DonutDataset(desktop_dataset_dict,
                               max_length=max_length,
                               split="validation",
                               task_start_token="<s_refexp>",
                               prompt_end_token="<s_target_bounding_box>",
                               sort_json_key=False,
                               )

    print(f"train dataset length: {train_dataset.dataset_length}")
    print(f"validation dataset length: {val_dataset.dataset_length}")

    torch.cuda.empty_cache()

    #@title Set optimal batch size for training and validation
    # Currently there is an issue with VisualEncoderDecoder when batch size > 1
    # Causes error in loss calculation during training

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


    def getPLModuleConfig():
      config = {"max_epochs": max_epochs_per_loop, # aim for 30,
                "val_check_interval": 1.0, # how many times we want to validate during an epoch
                "check_val_every_n_epoch":1,
                "gradient_clip_val":1.0,
                "num_training_samples_per_epoch": num_training_samples_per_epoch,
                "lr": learning_rate,
                "train_batch_sizes": [train_batch_size],
                "val_batch_sizes": [val_batch_size],
                # "seed":2022,
                # "num_nodes": 1,
                "warmup_steps": warmup_steps,
                "result_path": "./result",
                "verbose": True,
                }
      print(f'PL Module Config: {config}')
      return config


    # Take advantage of A100 GPU features
    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision('medium')


    def prep_trainer():
      global processor, model, trainer, model_module
      config = getPLModuleConfig()
      model_module = DonutModelPLModule(config, processor, model)
      torch.set_float32_matmul_precision('medium')

      lr_monitor = LearningRateMonitor()

      trainer = pl.Trainer(
              # accelerator="auto",
              # devices="auto",
              accelerator="gpu",
              devices=1,
              max_epochs=config.get("max_epochs"),
              val_check_interval=config.get("val_check_interval"),
              check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
              gradient_clip_val=config.get("gradient_clip_val"),
              precision=16, # we'll use mixed precision
              num_sanity_val_steps=0,
    #           logger=mlflow_logger,
              benchmark=True, # usually speeds up training,
              # strategy="ddp_notebook",
              # Other effective optimization techniques follow
              # https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html#accumulate-gradients
              # https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html#stochastic-weight-averaging
              # accumulate_grad_batches={0: 8}, # , 3: 4, 6: 8, 9: 4, 12: 2, 15: 1},
              callbacks=[lr_monitor], # , StochasticWeightAveraging(swa_lrs=1e-5)]
              # strategy="ddp_notebook",
              # callbacks=[lr_callback, checkpoint_callback],
      )
      # Create a ProgressBar callback and add it to the trainer
      progress_bar = ProgressBar()
      trainer.callbacks.append(progress_bar)


    prep_trainer()
    trainer.fit(model_module) # , ckpt_path="last")
