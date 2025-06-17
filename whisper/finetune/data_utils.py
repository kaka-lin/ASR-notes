# data_utils.py
from functools import partial
from typing import Dict
from transformers import WhisperProcessor

MAX_INPUT_LENGTH = 30.0  # 最大輸入長度（秒），根據 Whisper 模型的限制


def prepare_dataset(example: Dict, processor: WhisperProcessor) -> Dict:
    """
    將 Common Voice 資料集的音訊和文字轉換為 Whisper 模型的輸入格式。
    Args:
        example (dict): Common Voice 資料集中的一個樣本。
        processor (WhisperProcessor): Whisper 模型的處理器。
    Returns:
        dict: 處理後的樣本，包含音訊和文字的處理結果。
    """
    # 取得音訊資料
    audio = example["audio"]

    # 使用 WhisperProcessor 處理音訊和文字
    # returns a dictionary with input_features and labels
    # input_features: the audio features for the model
    # labels: the tokenized text for the model
    example = processor(
        audio=audio["array"],
        sampling_rate=audio["sampling_rate"],
        text=example["sentence"],
    )

    # compute input length of audio sample in seconds
    example["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    return example


def is_audio_in_length_range(length: float) -> bool:
    return length < MAX_INPUT_LENGTH
