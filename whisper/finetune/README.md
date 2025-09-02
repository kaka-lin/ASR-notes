# Fine-tuning Whisper Model

> [!important]
> 詳細的微調步驟與說明，請參考：[Fine-tuning Whisper Model Step by Step](./finetune_step_by_step.md)

## Prepare Environment

- `datasets[audio]`: 資料下載與準備
- `transformers` & `accelerate`: 模型載入與訓練
- `soundfile`: 音訊檔案前處理
- `evaluate` & `jiwer`: 評估模型效能

```bash
$ pip install -r requirements.txt
```

## Quickly Start

```bash
$ python finetune.py
```

For open TensorBoard

```bash
$ tensorboard --logdir=./logs

# assign port
$ tensorboard --logdir=./logs --port 6008
```
