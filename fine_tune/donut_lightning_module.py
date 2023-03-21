from pathlib import Path
import re
# from nltk import edit_distance
import numpy as np
import math

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class DonutModelPLModule(pl.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = train_batch_size
        self.learning_rate = self.config.get("lr")
        # self.log_dict(config)

    def training_step(self, batch, batch_idx):
        pixel_values, decoder_input_ids, labels = batch

        outputs = self.model(pixel_values,
                             decoder_input_ids=decoder_input_ids[:, :-1],
                             labels=labels[:, 1:])
        loss = outputs.loss
        self.log_dict({"train_loss": loss}, sync_dist=True)
        return loss

    def token2bbox(self, seq: str):
        target_bbox = self.processor.token2json(seq)
        bbox = target_bbox.get('target_bounding_box')
        if bbox is None:
            print(f"token2bbox seq has no target_bounding_box, seq:{seq}")
            bbox = bbox = {"xmin": 0, "ymin": 0, "xmax": 0, "ymax": 0}
            return bbox
        # print(f"token2 bounding box json: {bbox}")
        # safeguard in case text prediction is missing some bounding box coordinates
        # or coordinates are not valid numeric values
        try:
            xmin = float(bbox.get("xmin", 0))
        except Exception:
            xmin = 0
        try:
            ymin = float(bbox.get("ymin", 0))
        except Exception:
            ymin = 0
        try:
            xmax = float(bbox.get("xmax", 1))
        except Exception:
            xmax = 1
        try:
            ymax = float(bbox.get("ymax", 1))
        except Exception:
            ymax = 1
        # replace str with float coords
        bbox = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
        # print(f"token2 bounding box float: {bbox}")
        return bbox

    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pixel_values, decoder_input_ids, prompt_end_idxs, answers = batch
        decoder_prompts = pad_sequence(
            [input_id[: end_idx + 1] for input_id, end_idx in zip(decoder_input_ids, prompt_end_idxs)],
            batch_first=True,
        )

        outputs = self.model.generate(pixel_values,
                                      decoder_input_ids=decoder_prompts,
                                      max_length=max_length,
                                      early_stopping=True,
                                      pad_token_id=self.processor.tokenizer.pad_token_id,
                                      eos_token_id=self.processor.tokenizer.eos_token_id,
                                      use_cache=True,
                                      num_beams=1,
                                      bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                                      return_dict_in_generate=True, )

        predictions = []
        for seq in self.processor.tokenizer.batch_decode(outputs.sequences):
            seq = seq.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            predictions.append(seq)

        scores = list()
        for pred, answer in zip(predictions, answers):
            answer = re.sub(r"<.*?>", "", answer, count=1)
            answer = answer.replace(self.processor.tokenizer.eos_token, "")
            answer_bbox = self.token2bbox(answer)
            pred_bbox = self.token2bbox(pred)
            # scores.append(get_center_distance(pred_bbox, answer_bbox))
            # scores.append(get_iou(pred_bbox, answer_bbox))
            scores.append(get_cui(pred_bbox, answer_bbox))
            if self.config.get("verbose", False) and len(scores) == 1:
                print(f"      Prediction: {pred}")
                print(f"          Answer: {answer}")
                print(f" Prediction bbox: {pred_bbox}")
                print(f"     Answer bbox: {answer_bbox}")
                # print(f"Eval score (Center Distance): {scores[0]}")
                print(f"Eval score CUI=CDx(U-I): {scores[0]}")
                iou = get_iou(pred_bbox, answer_bbox)
                print(f"Eval score (IoU): {iou}")
                # print(f"Eval score (IoU): {scores[0]}")
                # print(f"Eval score (Edit Distance): {scores[2]}")

        return scores

    def on_validation_epoch_end(self, validation_step_outputs):
        # I set this to 1 manually
        # (previously set to len(self.config.dataset_name_or_paths))
        num_of_loaders = 1
        if num_of_loaders == 1:
            validation_step_outputs = [validation_step_outputs]
        assert len(validation_step_outputs) == num_of_loaders
        cnt = [0] * num_of_loaders
        total_metric = [0] * num_of_loaders
        val_metric = [0] * num_of_loaders
        for i, results in enumerate(validation_step_outputs):
            for scores in results:
                cnt[i] += len(scores)
                total_metric[i] += np.sum(scores)
            val_metric[i] = total_metric[i] / cnt[i]
            val_metric_name = f"val_metric_{i}th_dataset"
            self.log_dict({val_metric_name: val_metric[i]}, sync_dist=True)
        self.log_dict({"val_metric": np.sum(total_metric) / np.sum(cnt)}, sync_dist=True)

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html?highlight=configure_optimizers#configure-optimizers
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        # we use max below, because we want the lr to decrease if IoU stops increasing
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2)  # previously patience=3, 5
        # log initial value for val_metric to avoid train error before its calculated
        # self.log_dict({"val_metric": 0.0}, sync_dist=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_metric",  # track IoU progress
                # "frequency": "indicates how often the metric is updated"
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            }
        }

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader