# ASR-notes

A collection of notes, tutorials, and implementations for Automatic Speech Recognition (ASR).
Covers fundamentals, popular open-source models (like Whisper), and practical use cases such as real-time transcription and model fine-tuning.

---

## Contents

### 1. ASR 基礎 (Fundamentals)

- **[音訊資料處理 (Audio Data)](./basic/audio_data/README.md)**: 介紹波形、頻譜、梅爾頻譜等基本概念。
- **[評估指標 (Metrics)](./basic/metrics/metric.md)**: 解釋 WER, CER 等常用於評估 ASR 模型效能的指標。

### 2. 模型與架構 (Models & Architectures)

- **[核心架構：串流 vs. 離線 (Streaming vs. Offline)](./streaming_asr_technical_deep_dive.md)**: 深入解析兩種基礎 ASR 架構的原理、優缺點與應用場景。

- **模型解析：Whisper**
  - **[Whisper 模型介紹](./whisper/README.md)**: 包含 Whisper 的模型結構、特點與基本使用。
  - **[將 Whisper 改造為串流模式](./streaming_asr_technical_deep_dive.md#核心議題將-whisper-離線模型改造為串流模式)**: 探討將 Whisper 從離線模型改造成即時串流的幾種主流技術路線。
  - **[模型微調 (Fine-tuning)](./whisper/finetune/finetune_step_by_step.md)**: 提供逐步指南，說明如何對 Whisper 進行微調以適應特定領域的資料。

### 3. 雲端 ASR 服務 (Cloud-based ASR APIs)

- **Google Cloud Speech-to-Text V2**
  - **Overview**: [Product page](https://cloud.google.com/speech-to-text)
  - **Docs**: [Official documentation](https://cloud.google.com/speech-to-text/docs/)
  - **Features**: `chirp_2` model, batch/streaming, multi-language support.
  - **Toolkit**: 🧰 [gcloud-python-toolkit](https://github.com/kaka-lin/gcloud-python-toolkit) - A collection of Python scripts for transcribing audio using the `chirp_2` model.

### 4. 相關工具與專案 (Related Tools & Repositories)

- 🔊 **[Multi-ASR Toolkit](https://github.com/kaka-lin/multi-asr-toolkit)**: A command-line and Web UI interface for speech recognition apps using Whisper or SpeechRecognition.
- 🧰 **[audio-tools](https://github.com/kaka-lin/audio-tools)**: Utilities for working with audio: WAV reader/writer, recording, ALSA/tinyalsa wrappers.
- 📊 **[audio-analysis-tools](https://github.com/kaka-lin/audio-analysis-tools)**: Tools for spectral analysis, FFT visualization, and feature extraction.
- 😊 **[speech-emotion-recognition](https://github.com/kaka-lin/speech-emotion-recognition)**: Deep learning models for detecting emotion from audio.
