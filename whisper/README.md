# Whisper Notes

這個目錄包含了 Whisper 模型的相關筆記、實作與效能評估。

## 目錄

- **Whisper**
  - [Whisper 簡介](./whisper_introduction.md): Whisper 模型架構、預訓練目標與相關資源的詳細介紹。

  - [Fine-tuning Whisper Model](./finetune/README.md): 如何使用自己的資料集對 Whisper 模型進行微調的步驟與說明。

  - [Whisper Model Benchmark](./benchmark/README.md): 針對不同 Whisper 模型的速度與記憶體佔用進行的效能測試。

- **Faster Whisper**
  - [faster-whisper 效能調校](./faster_whisper_thread.md): `faster-whisper` 中 `cpu_threads` 與 `num_workers` 參數的說明與比較。
