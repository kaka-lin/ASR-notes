import logging
from threading import Thread
from typing import Optional

from faster_whisper import WhisperModel


def get_model(model_name: str, compute_type: str, device_index: int) -> WhisperModel:
    """
    根據給定的模型名稱、計算類型和設備索引獲取 Whisper 模型。
    """
    print(f"正在載入模型 '{model_name}'，計算類型為 '{compute_type}'...")
    model = WhisperModel(
        model_name,
        device="cuda",
        device_index=device_index,
        compute_type=compute_type,
    )
    print("模型載入完成。")
    return model


def inference(model: WhisperModel, audio_path: str):
    """
    使用給定的模型執行一次完整的推論任務。
    """
    print(f"正在對 '{audio_path}' 進行推論...")
    segments, info = model.transcribe(audio_path, language="fr", beam_size=5)
    # 迭代 segments 以確保所有計算都完成
    count = 0
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        count += 1
    print(f"推論完成，共處理 {count} 個片段。")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    設定並返回一個 logger。
    """
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class MyThread(Thread):
    """
    用於在背景執行函式的自訂執行緒類別。
    """
    def __init__(self, func, params):
        super(MyThread, self).__init__()
        self.func = func
        self.params = params
        self.result = None

    def run(self):
        self.result = self.func(*self.params)

    def get_result(self):
        return self.result
