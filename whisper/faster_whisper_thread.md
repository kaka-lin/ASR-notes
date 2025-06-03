# faster-whisper: cpu_threads 與 num_workers 說明

本文件介紹 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 中兩個重要參數 `cpu_threads` 與 `num_workers` 的用法與差異，並提供在不同執行模式（CPU、GPU）下的範例。

## Whisper 推理流程簡介

> 音訊前處理 (Mel Spectrogram) -> Encoder -> Decoder

Whisper 的推理流程主要分為三個階段：

1. 音訊前處理 (CPU): 將 Audio 轉換為 [Mel Spectrogram](https://github.com/kaka-lin/ASR-notes/blob/main/basic/audio_data/introduction.md#7-mel-spectrogram%E6%A2%85%E7%88%BE%E9%A0%BB%E8%AD%9C%E5%9C%96) 特徵
   - 完全在 CPU 上執行。
   - 可以使用 `cpu_threads` 控制此階段的多執行緒加速。

2. 模型編碼 (Encoder): 將 Mel 特徵轉換為高維隱藏向量
    - 在 CPU 模式下使用 `cpu_threads`，GPU 模式則交由 GPU 處理。

3. 自回歸解碼 (Decoder): 根據 encoder 輸出逐步生成文字 token
    - 為自回歸性質，此階段較耗時。
    - 在 CPU 模式下使用 `cpu_threads`，GPU 模式則交由 GPU 處理。

## 參數概述

- **`cpu_threads`**: 控制每個推理任務中，在 CPU 上使用多少執行緒 (thread) 處理前處理與模型推理 (限 CPU 模式)
  - 影響音訊前處理 (Mel Spectrogram 轉換)。
  - 若模型跑在 CPU 上，也影響 Encoder/Decoder 運算的多執行緒加速。

- **`num_workers`**: 控制在同一個 CTranslate2 實例下，可同時啟動多少個獨立 Worker (工作單元) 去承接不同的推理請求
  - 每個 Worker 底層都依照 `cpu_threads` 開啟對應數量的執行緒，並行處理整條推理流程。
  - 主要用於「多條請求同時到達」時，在 CPU 模式下提高併發吞吐。
  - 在 GPU 模式下通常不會生效，因為都是由 GPU 自身調度。

---

## 在 CPU 模式下的行為

當模型 `device="cpu"` 時，呼叫 `model.transcribe()`後`cpu_threads` 與 `num_workers` 行爲如下:

### cpu_threads

1. **範圍**
   - 音訊前處理 (Mel spectrogram)。
   - 模型編碼器與解碼器 (Encoder/DecodeR) 推理。

2. **作用**
   - **特徵擷取階段** (Audio → Mel Spectrogram): 若 `cpu_threads=8`，那麼壓縮長音檔的運算會拆成 8 份同時處理。
   - **模型運算階段** (Encoder + Decoder): 矩陣乘法、注意力計算等也會依設定的執行緒數切分成多個任務並行執行。

    > `cpu_threads` 決定 CTranslate2 對這些步驟如何使用多條 CPU 執行緒並行計算（通常透過 OpenMP 或類似技術）。

1. **適用情況**
   - 當你需要加速「單條請求的速度」。
   - 通常建議設定為物理核心數（或物理核心數×2）如 4、8。

### num_workers

> 每個 Worker 就像一組完整的推理執行環境，負責拿到一條音訊請求後執行——從特徵擷取、Encoder 到 Decoder，一路跑到底。

1. **範圍**
   - 控制同一個 CTranslate2 實例裡能同時有多少個 Worker 在跑「完整推理流程」。
   - 若 `num_workers=2`，CTranslate2 會預先啟動 2 個 Worker，分別等待不同請求。

2. **作用**
   - **多請求併發**：當程式同時呼叫兩次 `transcribe()`，如果 Worker1 正在忙，第二條請求就由 Worker2 處理，互不排隊。
   - 若請求數量超過 Worker 數目，多餘請求會排隊等待空閒 Worker。
   - 每個 Worker 內部再透過 `cpu_threads` 控制自身的多執行緒運算。

3. **適用情況**
   - 當後端伺服器上有多個推理請求，如: 多個用戶呼叫 `transcrbie()`。
   - 例如：`num_workers=2, cpu_threads=4` → 最多兩條請求並行，每條請求內部用四個執行緒加速。

### CPU 模式範例

```python
import threading
from faster_whisper import WhisperModel

# 建立一個 WhisperModel，指定 CPU 推理
model = WhisperModel(
    model_size_or_path="large-v3",
    device="cpu",
    cpu_threads=4,    # 單條請求內部使用 4 執行緒做特徵擷取與解碼
    num_workers=2     # 最多同時兩個 Worker 處理不同請求
)

# 假設有兩個音訊檔要同時轉寫
audio_paths = ["audio1.wav", "audio2.wav"]

def transcribe_file(path):
    result = model.transcribe(path)
    text = "".join([seg.text for seg in result])
    print(f"[{path}] 轉寫結果：\n{text}\n")

threads = []
for p in audio_paths:
    t = threading.Thread(target=transcribe_file, args=(p,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

## 在 GPU 模式下的行為

當模型 `device="cuda"` 時，呼叫 `model.transcribe()`後`cpu_threads` 與 `num_workers` 行爲如下:

### cpu_threads (在 GPU 模式下)

1. **範圍**
   - 即使模型跑在 GPU 上，Whisper 的「音訊前處理階段 (Mel Spectrogram)」仍在 CPU 執行。
   - `cpu_threads` 控制 CTranslate2 在這段前處理中所使用的 CPU 執行緒數。

2. **作用**
   - 在前處理時若 `cpu_threads=4`，就能用 4 條執行緒並行計算 Mel Spectrogram。
   - 一旦特徵計算完畢，才把資料丟給 GPU 進行 Encoder + Decoder 推理；此階段由 GPU 自行排程，不受 `cpu_threads` 影響

3. **適用情況**
   - 若音訊前處理本身耗時較長（如長時間雜訊濾波、大量重採樣），調高 `cpu_threads` 能有效縮短前處理時間。
   - 若前處理短且 GPU 推理已成瓶頸，可將 `cpu_threads` 保持預設 (1-2)。

### num_workers (在 GPU 模式下)

> 每個 Worker 就像一組完整的推理執行環境，負責拿到一條音訊請求後執行——從特徵擷取、Encoder 到 Decoder，一路跑到底。

1. **範圍**
   - 設定多個 Worker，但在單卡 GPU 上，CTranslate2 並不會為每個 Worker 各自分配不同的 GPU 執行串流。
   - 所有進來的推理請求都被送到同一張 GPU，GPU 會將 kernel launch 排到同一個命令佇列中。

2. **作用**
   - 設定 `num_workers>1` 只是讓底層多個 Worker 同時等待請求，但最終所有推理排到同一張 GPU。
   - GPU 已有自己的併行架構（多重 SM、CUDA stream），因此無論有多少 Worker，把任務丟進同一張卡，併行度不會因為 Worker 數量增加而提升。

3. **適用情況**
   - **通常不必改動**: 在單卡 GPU 模式下，把 `num_workers` 保持預設 (1) 即可。
   - 若要透過多張 GPU 並行，必須為各張卡建立不同模型實例，如:
     - `device="cuda:0"`
     - `device="cuda:1"`

### GPU 模式範例

```python
import threading
from faster_whisper import WhisperModel

# 建立一個 WhisperModel，指定用 GPU 推理
model = WhisperModel(
    model_size_or_path="large-v3",
    device="cuda",    # Encoder/Decoder 一律送到 GPU 執行
    cpu_threads=4,    # 影響音訊前處理，使用 4 條 CPU 執行緒
    num_workers=1     # GPU 模式下 num_workers 對推理無明顯影響，設 1 即可
)

# 假設有三個音訊檔要同時轉寫
audio_paths = ["audio1.wav", "audio2.wav", "audio3.wav"]

def transcribe_file(path):
    result = model.transcribe(path)
    text = "".join([seg.text for seg in result])
    print(f"[{path}] 轉寫結果：\n{text}\n")

threads = []
for p in audio_paths:
    t = threading.Thread(target=transcribe_file, args=(p,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

### 多張 GPU 時分派方式

若機器有多張 GPU 且希望將請求分散到不同卡上，可為每張卡建立獨立的模型實例，並在 Thread 或進程中指定不同 device。例如:

```python
import threading
from faster_whisper import WhisperModel

def transcribe_on_gpu0(path):
    model = WhisperModel(
        model_size_or_path="large-v3",
        device="cuda:0",  # 模型放在 GPU 0
        cpu_threads=4,
        num_workers=1
    )
    result = model.transcribe(path)
    text = "".join([seg.text for seg in result])
    print(f"[GPU0][{path}] 轉寫結果：\n{text}\n")

def transcribe_on_gpu1(path):
    model = WhisperModel(
        model_size_or_path="large-v3",
        device="cuda:1",  # 模型放在 GPU 1
        cpu_threads=4,
        num_workers=1
    )
    result = model.transcribe(path)
    text = "".join([seg.text for seg in result])
    print(f"[GPU1][{path}] 轉寫結果：\n{text}\n")

threads = []
# Thread A 使用 GPU:0
t1 = threading.Thread(target=transcribe_on_gpu0, args=("audio1.wav",))
# Thread B 使用 GPU:1
t2 = threading.Thread(target=transcribe_on_gpu1, args=("audio2.wav",))

t1.start()
t2.start()
threads.extend([t1, t2])

for t in threads:
    t.join()
```
