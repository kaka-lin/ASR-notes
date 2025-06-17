from functools import partial

import torch
import evaluate
from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from data_utils import prepare_dataset, is_audio_in_length_range
from collator import DataCollatorSpeechSeq2SeqWithPadding
from metrics import compute_metrics, load_metrics


def get_device_config():
    return torch.cuda.is_available()


def get_processor_and_model():
    # 1. Load the WhisperProcessor
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small", language="zh", task="transcribe"
    )

    # 2. Load the WhisperForConditionalGeneration model
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    # 3. Disable cache during training since it's incompatible with gradient checkpointing
    model.config.use_cache = False

    # 4. Set the language and task for generation and re-enable cache
    #   - `language` is used to set the language of the model
    #   - `task` is used to set the task of the model, e.g. "transcribe" or "translate"
    #   - `use_cache` is set to True to enable caching during generation
    model.generate = partial(
        model.generate,
        language="zh",
        task="transcribe",
        use_cache=True
    )

    return processor, model


def main():
    # 1. get dataset
    ds = DatasetDict({
      "train": load_dataset("mozilla-foundation/common_voice_13_0", "zh-TW", split="train+validation"),
      "test":  load_dataset("mozilla-foundation/common_voice_13_0", "zh-TW", split="test"),
    }).select_columns(["audio", "sentence"])

    # 2. Get processor and model
    processor, model = get_processor_and_model()

    # 3. Load metrics and config
    wer, cer = load_metrics()
    use_fp16 = get_device_config()
    sampling_rate = processor.feature_extractor.sampling_rate

    # 4. using `cast_column` to resample audio to the correct sampling rate
    ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

    # 5. Prepare dataset:: Convert audio and text to model input format
    # ds = ds.map(
    #     partial(prepare_dataset, processor=processor),
    #     remove_columns=ds.column_names["train"],
    #     num_proc=1
    # )

    # # 6. filter audio samples based on length
    # ds["train"] = ds["train"].filter(
    #     is_audio_in_length_range,
    #     input_columns=["input_length"],
    # )

    # 5+6. Map and filter the dataset
    # Note: This is a more efficient way to prepare the dataset
    #       by combining the mapping and filtering steps into one operation.
    ds["train"] = ds["train"].map(
        lambda ex: prepare_dataset(ex, processor),
        remove_columns=ds["train"].column_names,
        num_proc=1,
    ).filter(is_audio_in_length_range, input_columns=["input_length"])

    # 7. Prepare data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # 8. Prepare training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-tw",  # name on the HF Hub
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=50,
        max_steps=500,  # increase to 4000 if you have your own GPU or a Colab paid plan
        gradient_checkpointing=True,
        fp16=use_fp16,  # only use if you have a GPU with at least 16GB of VRAM
        fp16_full_eval=use_fp16,  # only use if you have a GPU with at least 16GB of VRAM
        evaluation_strategy="steps",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=500,
        eval_steps=500,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
    )

    # 9. Prepare trainer
    # Note: WhisperProcessor already includes a text normalizer, but we can use a basic one for simplicity
    #       if you want to customize the text normalization.
    #
    #   - normalizer is used to normalize the text, e.g. lowercasing, removing punctuation, etc.
    normalizer = BasicTextNormalizer()

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        data_collator=data_collator,
        compute_metrics=partial(
            compute_metrics, processor=processor, normalizer=normalizer, metric=wer
        ),
        tokenizer=processor,
    )

    # 10. Start training
    trainer.train()


if __name__ == "__main__":
    main()
