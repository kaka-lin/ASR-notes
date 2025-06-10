# ASR-notes

A collection of notes, tutorials, and implementations for Automatic Speech Recognition (ASR).
Covers fundamentals, popular open-source models (like Whisper, DeepSpeech), and practical use cases such as real-time transcription and model fine-tuning.
Also includes audio data handling and analysis tools.

---

## Contents

### Basic

- [Audio Data](./basic/audio_data/README.md)
- [Metrics](./basic/metrics/metric.md)

### Whisper

- [Whisper Notes](./whisper/README.md)

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

- 🧰 [gcloud-python-toolkit](https://github.com/kaka-lin/gcloud-python-toolkit)

  A collection of Python scripts and utilities for interacting with Google Cloud services, including:
    - **Google Drive Downloader**: List and download all files from a specified folder.
    - **Google Speech Transcribe**: Transcribe audio using the `chirp_2` model.

---

### Related Repositories

Practical tools and apps that extend ASR capabilities.

- 🔊 [Multi-ASR Toolkit](https://github.com/kaka-lin/multi-asr-toolkit)
  Command-line and Web UI interface for speech recognition apps using Whisper or SpeechRecognition.

- 🧰 [audio-tools](https://github.com/kaka-lin/audio-tools)
  Utilities for working with audio: WAV reader/writer, recording, ALSA/tinyalsa wrappers.

- 📊 [audio-analysis-tools](https://github.com/kaka-lin/audio-analysis-tools)
  Tools for spectral analysis, FFT visualization, and feature extraction.

- 😊 [speech-emotion-recognition](https://github.com/kaka-lin/speech-emotion-recognition)
  Deep learning models for detecting emotion from audio, based on datasets like RAVDESS.
