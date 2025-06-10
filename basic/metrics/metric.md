# Evaluation metrics for ASR

在評估語音辨識系統時，我們會將系統的預測結果與目標的文字轉錄（參考文本）進行比對，並標註其中出現的錯誤。我們將這些錯誤分為三種類型：

- 替換 (Substitution, S): 預測結果中的詞彙與正確詞彙不符，例如將「sat」聽成「sit」。
- 插入 (Insertion, I): 在預測中多加了一個本不應出現的詞彙。
- 刪除 (Deletion, D): 在預測中遺漏了應該出現的詞彙。

這些錯誤分類在所有語音辨識的評估指標中都是一致的。不同之處在於我們計算錯誤的層級：可以在`「詞」的層級（word-level)` 進行計算，也可以在`「字元」的層級（character-level）`進行。

接下來我們會用一個範例來說明每種評估指標的定義。在這個例子中，我們有一段正確的參考文本：

```
reference = "the cat sat on the mat"
```
我們正在嘗試評估的語音辨識系統的預測序列：
```
prediction = "the cat sit on the"
```

如上所示，我們可以看到，預測結果相當接近，但有些字不太準確。我們將根據三個最受歡迎的語音辨識指標來評估這個預測，看看每個指標會得到什麼樣的結果。

## 詞錯誤率 (Word Error Rate, WER)

> $WER=\dfrac{S+I+D}{N}$

> - 拼字錯誤無論大小都會被完整懲罰，只要詞與詞不完全一致，就算錯

詞錯誤率 (WER) 是語音辨識領域最常使用的指標。它會根據「詞」的層級來計算錯誤的數量，包括：

- 替換（Substitutions）
- 插入（Insertions）
- 刪除（Deletions）

這表示錯誤是以每個詞為單位進行標註的。舉個例子：

| Reference: | the | cat | sat | on | the | mat |
| -- | -- | -- | -- | -- | -- | -- |
| Prediction: | the | cat | sit | on | the |
| Label: | ✅ | ✅ | S | ✅ | ✅ | D |

在這個例子中我們有：

- 1 個替換錯誤: 將 "sat" 聽成 "sit"
- 0 個插入錯誤
- 1 個刪除錯誤: "mat" 遺漏了

總共是 2 個錯誤。接著，我們將錯誤數除以參考句中的總詞數 (N) 來計算錯誤率，這邊是 6：

$$WER=\dfrac{S+I+D}{N}=\dfrac{1+0+1}{6}=0.333$$

也就是說，*WER = 0.333 or 33%*。值得注意的是：「sit」這個詞雖然只錯了一個字母，但整個詞就被判定為錯誤。這正是 `WER 的特點之一：拼字錯誤無論大小都會被完整懲罰，只要詞與詞不完全一致，就算錯`。

### 關於 WER 的幾個重點

- WER 越低代表辨識效果越好。
- 完美的語音辨識系統 WER 應該是 0（表示沒有錯誤）。

### 使用 [Evaluate](https://github.com/huggingface/evaluate) 套件計算 WER

我們可以使用 Hugging Face 的 [evaluate](https://github.com/huggingface/evaluate) 搭配 [jiwer](https://github.com/jitsi/jiwer) 這兩個套件來計算 WER：

```bash
pip install --upgrade evaluate jiwer
```

然後

```python
from evaluate import load

wer_metric = load("wer")

wer = wer_metric.compute(references=[reference], predictions=[prediction])

print(wer)
```

輸出：

```bash
0.3333333333333333
```

### 一個常見的誤解：WER 的上限是多少？

你可能會以為 WER 的上限是 1（或 100%），對吧？其實不是！
因為 `WER 是錯誤數量除以參考詞數（N），所以它是沒有上限的`！

舉個例子：

如果參考句只有 2 個詞，但預測句中有 10 個完全錯誤的詞，而且所有預測都錯了（10個錯誤），那麼 WER 如下：

$$WER=\dfrac{10}{2}=5.0\Rightarrow500\%$$

所以，如果你在訓練 ASR 系統時發現 WER 超過 100%，這是可能的，但很可能代表模型出了點狀況。

## 詞準確率 (Word Accuracy, WAcc)

我們可以*將 WER（詞錯誤率）反過來表示，變成一個「越高越好」的指標*。也就是說，與其衡量錯誤率，我們也可以衡量系統的*詞準確率（WAcc）*：

$$WAcc=1−WER$$

WAcc 一樣是以「詞」為單位來計算，只不過它是從錯誤率（WER）轉換成準確率的形式。

不過，在語音辨識領域的研究文獻中，WAcc 很少被引用或使用。這是因為我們通常會從「錯誤」的角度來看待系統的預測結果，也就是偏好使用與錯誤類型標註（替換、插入、刪除）更密切相關的錯誤率指標（如 WER），而不是準確率指標。

## 字元錯誤率（Character Error Rate, CER）

> $CER=\dfrac{S+I+D}{N}$
> - CER 的優點： 對於拼字錯誤的懲罰比較溫和，更細緻地反映預測的正確程度。

將整個「sit」這個詞標成錯誤，看起來有點不太公平，因為實際上只有一個字母錯了。這是因為我們在進行詞級（word-level）評估，也就是以整個詞為單位來標註錯誤。

字元錯誤率（CER） 則是以「字元」為單位進行評估。也就是說，我們會把每個詞拆成一個一個的字元，然後逐字標示錯誤，如下有兩種方式：

---

#### 1. 不考慮空格及標點符號

| Reference: | t | h | e | | c | a | t | | s | a | t | | o | n | | t | h | e | | m | a | t |
| ----------- | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| Prediction: | t | h | e | | c | a | t | | s | i | t | | o | n | | t | h | e | |  |  |  |
| Label: | ✅ | ✅ | ✅ | | ✅ | ✅ | ✅ | | ✅ | S | ✅ | | ✅ | ✅ | | ✅ | ✅ | ✅ | | D | D | D |

現在我們可以看到，「sit」這個詞中，「s」和「t」被標為正確，只有中間的「i」是錯的（替換錯誤）。因此，我們獎勵了我們的系統部分正確的預測。

在這個例子中，我們有：

- 1 個字元替換錯誤: "i"
- 0 個插入錯誤
- 3 個刪除錯誤: "mat" 被省略

整段參考句總共有 17 個字元，所以我們的 CER 為：

$$CER=\dfrac{S+I+D}{N}=\dfrac{1+0+3}{17}=0.235$$

也就是說，*CER = 0.235 or 23.5%*

#### 2. 考慮空格及標點符號

| Reference: | t | h | e | | c | a | t | | s | a | t | | o | n | | t | h | e | | m | a | t |
| ----------- | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| Prediction: | t | h | e | | c | a | t | | s | i | t | | o | n | | t | h | e | |  |  |  |
| Label: | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | S | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | D | D | D | D |

現在我們可以看到，「sit」這個詞中，「s」和「t」被標為正確，只有中間的「i」是錯的（替換錯誤）。因此，我們獎勵了我們的系統部分正確的預測。

在這個例子中，我們有：

- 1 個字元替換錯誤: "i"
- 0 個插入錯誤
- 4 個刪除錯誤: "mat" 及 一個空格被刪除了

整段參考句總共有 22 個字元（含空格），所以我們的 CER 為：

$$CER=\dfrac{S+I+D}{N}=\dfrac{1+0+4}{22}=0.227$$

也就是說，*CER = 0.227 or 22.7%*

---

不管是哪一種方法，都可以看到這個 CER 數值比剛剛的 WER 還低，因為我們沒有將整個拼字錯誤的詞全部判為錯，而是根據實際錯誤的字元數來計算。這就是 `CER 的優點：
對於拼字錯誤的懲罰比較溫和，更細緻地反映預測的正確程度。`

### 使用 [Evaluate](https://github.com/huggingface/evaluate) 套件計算 CER

```python
from evaluate import load

cer_metric = load("cer")

# CER 預設是以 包含空格的完整字串長度 為母數計算
cer = cer_metric.compute(references=[reference], predictions=[prediction])

print(cer)
```

輸出：

```bash
0.22727272727272727
```
