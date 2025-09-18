# ASR 技術架構解析：三大核心模型與兩種服務模式

本文旨在解析當前主流的 ASR (Automatic Speech Recognition) 技術。我們將從三個核心的模型架構（CTC、Transducer、Attention Seq2Seq）出發，比較其內部機制與優劣，接著再探討這些模型如何被應用於兩種主要的服務架構——**離線 (Offline)** 與 **串流 (Streaming)**，以滿足不同的業務需求。

## 1. ASR 三大核心模型架構

當前 ASR 領域主要由三種主流模型架構主導，它們在處理語音到文字的對齊方式、延遲與建模能力上各有千秋。

📊 **CTC vs. Transducer vs. Attention Seq2Seq (以 Whisper 為例)**

| 特性 | CTC (Connectionist Temporal Classification) | Transducer (RNN-T / Conformer-Transducer) | Attention Seq2Seq (Whisper 等) |
| :--- | :--- | :--- | :--- |
| **主要組件** | Encoder + CTC head | Encoder + Prediction network + Joint network | Encoder + Decoder (Transformer/LSTM) |
| **對齊方式** | CTC loss：透過插入 `blank` 標籤進行單調對齊。 | Transducer loss：邊生成邊對齊，允許 `blank` 與遞增輸出。 | 透過 Attention 機制直接關注輸入的全局特徵，無明確 `blank`。 |
| **流式推理** | ✅ (天然支援) | ✅ (原生設計) | ❌ (需要額外改造才能實現) |
| **訓練難度** | 相對簡單；基於單調對齊假設。 | 中等；需要同步訓練 Encoder、Prediction 和 Joint 三個網絡。 | 較高；需要防止 Attention 機制在長序列上失效。 |
| **延遲** | 低 | 低至中 (取決於 Encoder 結構) | 高 (因需處理完整序列) |
| **建模能力** | 僅具備聲學建模能力，語言模型能力弱，需外接 LM。 | 內建簡易語言模型 (Prediction Network)，可與外部 LM 進行淺層融合。 | 擁有強大的上下文語言模型 (Decoder 自回歸機制)。 |
| **代表模型** | DeepSpeech v2 (部分)、Wenet-CTC、Kaldi CTC recipe | RNN-T、Conformer-Transducer、Emformer-Transducer、Chirp | Whisper、LAS (Listen-Attend-Spell)、SpeechTransformer |
| **適合場景** | 簡單流式任務、邊緣設備運算。 | 大多數商用 ASR 服務 (Google STT, Azure, AWS Transcribe)。 | 高準確率、大上下文需求的離線轉錄任務。 |

#### 總結

- **CTC**：結構簡單、速度快，但語言建模能力有限。
- **Transducer**：為流式辨識而生，是商用 ASR 的主流選擇，有效平衡了延遲與精度。
- **Attention Seq2Seq (Whisper)**：上下文建模能力最強，在離線轉錄場景中準確率頂尖。

---

## 2. 兩種 ASR 服務架構：離線 vs. 串流

上述的三大模型最終會被部署成以下兩種服務架構之一，以滿足不同的產品需求。

### 2-1. 離線架構 (Offline Architecture)

> [!Note]
> 將 `Whisper 離線模型改造為串流模式` 請看[這邊](./whisper/whisper_streaming_transformation.md)

離線架構以「準確性」為最高優先級，設計前提是模型必須在收到**完整的音訊片段**後才開始處理。

- **核心思想**: 先用 VAD (Voice Activity Detection) 偵測語音的起點與終點，待一段話結束後，將完整的音訊一次性送入 ASR 模型進行辨識。
- **適用模型**: `Attention Seq2Seq (Whisper)` 是此架構的典型代表，因為它需要全局上下文來發揮最大效能。
- **優點**:
  - **準確性高**: 模型能看到完整的上下文資訊，辨識準確率通常是最高的。
  - **工程實現簡單**: 邏輯單純，只需「偵測 -> 緩存 -> 辨識」。
- **缺點**:
  - **高延遲**: 必須等到使用者說完話並停頓後才能輸出結果，使用者會感到明顯的延遲。
  - **體驗不佳**: 缺乏即時回饋，無法在說話過程中看到文字。
- **應用場景**: 語音檔轉錄、會議記錄整理等對準確性要求高，但對即- **應用場景**: 語音檔轉錄、會議記錄整理等對準確性要求高，但對即時性要求不高的場景。

### 2-2. 串流架構 (Streaming Architecture)

串流架構為「即時性」而生，模型在結構上就被設計為逐塊處理音訊，並持續輸出中間結果。

- **核心思想**: 模型逐幀或逐個音訊塊地處理數據，並即時更新辨識結果，無需等待整句話結束。
- **適用模型**: `Transducer` 和 `CTC` 是此架構的原生支援者。
- **優點**:
  - **極低延遲**: 反應迅速，幾乎與語音同步輸出文字。
  - **互動體驗好**: 使用者可以即時看到回饋，並根據結果進行修正。
- **缺點**:
  - **準確性受限**: 由於缺乏未來上下文，準確性通常低於離線模型，且容易出現中間結果被修正的情況。
  - **斷句困難**: 需要額外的 VAD 邏輯來判斷一句話的真正結束。
- **應用場景**: 即時語音輸入、智慧助理、線上會議字幕等對即時性要求極高的場景。
