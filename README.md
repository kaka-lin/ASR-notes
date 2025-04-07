# ASR-notes

A collection of notes, tutorials, and implementations for Automatic Speech Recognition (ASR).
Convers fundamentails, popular open-sources model (like Whisper, DeepSpeech), and practical use cases such as real-time transcription and model fine-tuning.

---

## Contents

### Awesome Whisper

#### 1. Official Whisper

- Introduction: [Whisper on OpenAI](https://openai.com/index/whisper)
- GitHub: [openai/whisper](https://github.com/openai/whisper)
- Paper: [Robust Speech Recognition via Large-Scale Weak Supervision](https://cdn.openai.com/papers/whisper.pdf)

#### 2. Whisper Model variants

Whisper-based implementations that improve speed, compatibility, or functionality.

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): High-performance Whisper inference using CTranslate2 (CPU/GPU optimized)
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper): Whisper ported to Apple's MLX framework (runs on Apple Silicon)
- [WhisperX](https://github.com/m-bain/whisperX): Word-level timestamps and speaker diarization built on Whisper

#### 3. Whisper Streaming Implementations

Experimental or community-based projects that adapt Whisper for real-time/streaming ASR.

- [ufal/whisper_streaming](https://github.com/ufal/whisper_streaming): Real-time Whisper with buffering and chunked inference
- [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit): Local real-time STT with speaker diarization, FastAPI server & web UI

---

### Cloud-based ASR APIs

#### 1. Google Cloud Speech-to-Text V2

- Overview: [Product page](https://cloud.google.com/speech-to-text)
- Docs: [Official documentation](https://cloud.google.com/speech-to-text/docs/)
- Features:
  - Models: `chirp`, `chirp_2` (high accuracy)
  - Supports batch + streaming (`recognize`, `longRunningRecognize`, `streamingRecognize`)
  - Region-specific support (e.g., `us-central1`, `global`)
  - Languages: [Full list](https://cloud.google.com/speech-to-text/docs/languages)
  - Output formats: JSON, SRT, word-level timestamps, diarization, and punctuation

---

> ✨ Contributions welcome! This repo aims to be a well-organized reference for ASR experiments and deployment.
