# Whisper Model Benchmark

此目錄包含一系列用於評估 `faster-whisper` 模型在不同設定下的效能腳本，涵蓋了速度、記憶體佔用以及併發處理能力。

## 環境準備

### 1. 安裝依賴套件

首先，請安裝所有必要的 Python 套件：

```bash
pip install -r requirements.txt
```

### 2. 下載測試音訊

執行以下腳本，從 YouTube 下載用於基準測試的音訊檔案 (`benchmark.m4a`)：

```bash
python download_audio.py
```

## 使用方法

### 速度測試 (Speed Benchmark)

`speed_benchmark.py` 用於測量單次推論任務的執行速度。它會多次運行並回報最快的執行時間。

```bash
# 測試 large-v3 模型搭配 float16 精度的速度
python speed_benchmark.py --model large-v3 --compute_type float16
```

### 記憶體測試 (Memory Benchmark)

`memory_benchmark.py` 用於測量模型在載入和推論過程中的峰值記憶體使用量（支援 RAM 與 GPU VRAM）。

```bash
# 測量 large-v3 模型在 GPU 上的 VRAM 使用量
python memory_benchmark.py --model large-v3 --compute_type float16 --gpu_memory
```

### 並發與參數調校測試 (Concurrency & Tuning Benchmark)

`benchmark.py` 是一個綜合測試腳本，用於尋找在多請求併發情境下，`cpu_threads` 和 `num_workers` 參數的最佳組合。它會分別測試「共享模型 (Singleton)」與「獨立模型 (Non-Singleton)」兩種模式下的效能。

```bash
python benchmark.py
```

---

## 效能測試結果

### Large-v2 model on GPU

| Implementation | Precision | Beam size | Time | VRAM Usage |
| --- | --- | --- | --- | --- |
| openai/whisper | fp16 | 5 | 2m23s | 4708MB |
| whisper.cpp (Flash Attention) | fp16 | 5 | 1m05s | 4127MB |
| transformers (SDPA)[^1] | fp16 | 5 | 1m52s | 4960MB |
| faster-whisper | fp16 | 5 | 1m03s | 4525MB |
| faster-whisper (`batch_size=8`) | fp16 | 5 | 17s | 6090MB |
| faster-whisper | int8 | 5 | 59s | 2926MB |
| faster-whisper (`batch_size=8`) | int8 | 5 | 16s | 4500MB |

## Reference

- [faster-whisper/benchmark](https://github.com/SYSTRAN/faster-whisper/tree/master/benchmark)
