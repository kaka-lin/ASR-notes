import os
import sys
import pathlib
from typing import List, Dict

# 將專案根目錄加入模組搜尋路徑
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from dotenv import load_dotenv
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

from common.google_service import get_google_service
from google_speech_utils import create_speech_v2_client, create_recognizer

# 載入 .env 內容到環境變數
if not load_dotenv():
    print("警告：.env 檔案不存在或解析失敗，請確認它位於專案根目錄。")


def transcribe_file(
    speech_client: SpeechClient,
    audio_path: str,
    output_path: str,
    recongizer_name: str,
    language_codes: List[str] = ["cmn-Hant-TW"],
    model: str = "chirp_2",
) -> None:
    """
    使用 Google Cloud Speech-to-Text V2 API 轉錄音訊檔案。

    Args:
        speech_client (SpeechClient): Google Cloud Speech-to-Text V2 API 客戶端。
        audio_path (str): 要轉錄的音訊檔案路徑 (wav/flac/mp3...)。
        output_path (str): 轉錄結果輸出檔案的路徑。
        recongizer_name (str): full recognizer 名稱，例如 "projects/{project_id}/locations/{location}/recognizers/{recognizer_id}"。
        language_codes (List[str]): 語言代碼列表，預設為 ["cmn-Hant-TW"]。
        model (str): 語音識別模型，預設為 "chirp_2"。
    """
    if not isinstance(speech_client, SpeechClient):
        raise TypeError(f"speech_client 必須是 SpeechClient，實際收到 {type(speech_client)}")

    # 讀取音訊檔案
    audio_bytes = pathlib.Path(audio_path).read_bytes()

    # 設定辨識配置
    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        model=model,
        language_codes=language_codes,  # 明確指定語言代碼
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
            enable_word_time_offsets=True
        )
    )

    # 建立辨識請求
    request = cloud_speech.RecognizeRequest(
        recognizer=recongizer_name,
        config=config,
        content=audio_bytes
    )

    # 執行辨識
    response = speech_client.recognize(request=request)

    # Join the best alternative across all results
    lines: List[str] = []
    for result in response.results:
        if result.alternatives:
            lines.append(result.alternatives[0].transcript)
    transcribed_text = "\n".join(lines).strip()

    # 將轉錄結果寫入檔案
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcribed_text)
        print(f"✅ 轉錄完成：{output_path}")

    return transcribed_text


if __name__ == "__main__":
    # 讀取環境變數
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    GSPEECH_CREDENTIALS = os.getenv("GSPEECH_CREDENTIALS")
    if not GSPEECH_CREDENTIALS:
        print("錯誤：環境變數 GSPEECH_CREDENTIALS 未設定！")
        sys.exit(1)

    OUTPUT_DIR = os.getenv("DOWNLOAD_DIR", "downloads")
    AUDIO_DIR = os.path.join(OUTPUT_DIR, "audios")
    TRANSCRIBE_DIR = os.path.join(OUTPUT_DIR, "transcripts")
    os.makedirs(TRANSCRIBE_DIR, exist_ok=True)

    # 設置其他參數
    service_name = "speech"
    version = "v1"
    location = "us-central1"  # "us-central1" or "asia-southeast1"
    recognizer_id = "chirp-recognizer"
    language_codes = ["cmn-Hant-TW"]
    model = "chirp_2"
    SUPPORTED_FORMATS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")

    # 取得 Google Cloud Speech-to-Text API 服務物件及認證
    gspeech_service, gspeech_creds = get_google_service(
        service_name=service_name,
        version=version,
        credentials=GSPEECH_CREDENTIALS,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    # 創建 Google Cloud Speech-to-Text V2 API 客戶端
    speech_client = create_speech_v2_client(
        credentials=gspeech_creds,
        location=location
    )

    # 建立 recognizer
    recongizer = create_recognizer(
        speech_client=speech_client,
        project_id=PROJECT_ID,
        location=location,
        recognizer_id=recognizer_id,
        language_codes=language_codes,
        model=model
    )

    # 開始轉錄
    for audio_file in sorted(os.listdir(AUDIO_DIR)):
        print(f"處理音訊檔案：{audio_file}")
        if audio_file.lower().endswith(SUPPORTED_FORMATS):
            audio_path = os.path.join(AUDIO_DIR, audio_file)
            base_filename, _ = os.path.splitext(audio_file)
            transcript_path = os.path.join(TRANSCRIBE_DIR, f"{base_filename}.txt")
            text = transcribe_file(
                speech_client=speech_client,
                audio_path=audio_path,
                output_path=transcript_path,
                recongizer_name=recongizer.name,
                language_codes=language_codes,
                model=model
            )
            print(f"--- 轉錄結果 ({audio_file}) ---\n{text}\n")
    print("所有音訊檔案已轉錄完成！")
