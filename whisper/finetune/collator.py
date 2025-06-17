from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    It will pad the inputs to the maximum length in the batch, and will also pad the labels
    to the maximum length in the batch.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        # Huginface uses -100 as the ignore_index for the loss function.
        # Example: CrossEntropyLoss(ignore_index=-100)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )  # attention_mask not equal to 1 means padding token

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways.
        # Whisper model's decoder input will automatically add
        # the beginning of transcript token (BOS): "(<|startoftranscript|>)"
        # at the start of the sequence. So we need to remove it from the labels.
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
