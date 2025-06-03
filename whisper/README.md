# Whisper Notes

以下為 Whisper 系列的整理與分類

## Awesome Whisper

### 1. Official Whisper

- OpenAI 官方介紹: [Whisper on OpenAI](https://openai.com/index/whisper)
- GitHub: [openai/whisper](https://github.com/openai/whisper)
- Paper: [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)

### 2. Whisper Model Variants

Whisper-based implementations that improve speed, compatibility, or functionality.

| 名稱 | 特性 |
| --- | ---- |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper)                 | 使用 CTranslate2 加速推理（支援 CPU/GPU |
| [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) | Apple Silicon 上的 Whisper 實作 |
| [WhisperX](https://github.com/m-bain/whisperX)                              | 支援 Word-level timestamps 與 speaker diarization |

👉 延伸閱讀： [faster-whisper: cpu_threads 與 num_workers 說明](./faster_whisper_thread.md)

### 3. Whisper Streaming Implementations

Experimental or community-based projects that adapt Whisper for real-time/streaming ASR.

| 名稱 | 說明 |
| --- | ---- |
| [ufal/whisper\_streaming](https://github.com/ufal/whisper_streaming)                 | Real-time Whisper with buffering and chunked inference |
| [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit)                      | Local real-time STT with speaker diarization, FastAPI server & web UI |
| [whisper-streaming-practice](https://github.com/kaka-lin/whisper-streaming-practice) | Practical implementation of streaming Whisper inference with custom audio/VAD handling |
