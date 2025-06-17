from typing import Any, Dict
import evaluate


def compute_metrics(pred, processor: Any, normalizer: Any, metric: Any) -> Dict:
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 1. replace -100 with the pad_token_id
    #   - when calculating the loss, -100  means that the token should be ignored
    #   - To decode into text, we need to convert -100 back to the real `pad_token_id`,
    #     otherwise batch_decode will fail.
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # 2. we do not want to group tokens when computing the metrics
    # so we use `skip_special_tokens=True` to ignore special tokens like <|startoftranscript|>、<|endoftranscript|>、<|pad|>
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # 3. compute orthographic WER: including punctuation and upper/lower case
    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    # 4. compute normalised WER:
    # normalizer is used to normalize the text, e.g. lowercasing, removing punctuation, etc.
    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]

    # filtering step to only evaluate the samples that correspond to non-zero references:
    # we only want to compute WER for samples that have a non-empty reference.
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}


def load_metrics():
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")
    return wer, cer
