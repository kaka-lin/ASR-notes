import os
import sys
from typing import List

# 將專案根目錄加入模組搜尋路徑
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from dotenv import load_dotenv
from google.oauth2 import service_account
from google.cloud.speech_v2 import SpeechClient, Recognizer
from google.cloud.speech_v2.types import CreateRecognizerRequest
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import AlreadyExists

from common.google_service import get_google_service

# 載入 .env 內容到環境變數，並強制更新
if not load_dotenv(override=True):
    print("警告：.env 檔案不存在或解析失敗，請確認它位於專案根目錄。")


def create_speech_v2_client(
    credentials: service_account.Credentials,
    location: str = "us-central1"
) -> SpeechClient:
    """
    創建並返回 Google Cloud Speech-to-Text V2 API 的 API 客戶端。

    Args:
        credentials (Credentials): 已載入的 Service Account 憑證。
        location (str): API 端點區域 (e.g. 'us-central1' 或 'asia-southeast1')。

    Returns:
        SpeechClient: 可用於呼叫 recognizer 相關方法的客戶端實例。
    """
    # 創建 Google Cloud Speech-to-Text V2 API 客戶端
    return SpeechClient(
        credentials=credentials,
        client_options=ClientOptions(
            api_endpoint=f"{location}-speech.googleapis.com"
        )
    )


def create_recognizer(
    speech_client: SpeechClient,
    project_id: str,
    location: str,
    recognizer_id: str,
    language_codes: List[str],
    model: str
) -> Recognizer:
    """
    使用 Google Cloud Speech-to-Text V2 API 透過 Python 建立 recognizer。

    Args:
        speech_client (SpeechClient): Google Cloud Speech-to-Text V2 API 客戶端。
        project_id (str): GCP 專案 ID。
        location (str): 區域 (e.g., 'asia-southeast1')。
        recognizer_id (str): 要建立的 recognizer ID。
        language_codes (List[str]): 語言代碼列表，例如 ['cmn-Hant-TW']。
        model (str): 語音識別模型，例如 'chirp_2'。
    """
    if not isinstance(speech_client, SpeechClient):
        raise TypeError(f"speech_client 必須是 SpeechClient，實際收到 {type(speech_client)}")

    # 檢查參數
    parent = f"projects/{project_id}/locations/{location}"

    # 建立 RecognizerRequest
    recognizer = Recognizer(
        language_codes=language_codes,
        model=model
    )
    request = CreateRecognizerRequest(
        parent=parent,
        recognizer_id=recognizer_id,
        recognizer=recognizer
    )

    # 嘗試建立，若已存在則改用 get_recognizer
    try:
        operation = speech_client.create_recognizer(request=request)
        print("🛠️ Recognizer 建立中，請稍候...")
        response = operation.result()
        print(f"✅ Recognizer 已建立: {response.name}")
        return response
    except AlreadyExists:
        name = f"{parent}/recognizers/{recognizer_id}"
        print(f"⚠️ Recognizer `{recognizer_id}` 已存在，改為讀取現有資源：{name}")
        return speech_client.get_recognizer(name=name)


def list_us_central1_recognizers(
    speech_client: SpeechClient,
    project_id: str,
    location: str = "us-central1",
) -> None:
    """
    列出指定專案下的所有 recognizers。

    Args:
        speech_client (SpeechClient): Google Cloud Speech-to-Text V2 API 客戶端。
        project_id (str): GCP 專案 ID。
        location (str): 區域 (e.g., 'asia-southeast1')。
    """
    # 列出所有 recognizers
    parent = f"projects/{project_id}/locations/{location}"
    for rec in speech_client.list_recognizers(parent=parent):
        print(rec.name)


if __name__ == "__main__":
    # 讀取環境變數
    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")  # 從環境變數取得專案 ID

    GSPEECH_CREDENTIALS = os.getenv("GSPEECH_CREDENTIALS")
    if not GSPEECH_CREDENTIALS:
        print("錯誤：環境變數 GSPEECH_CREDENTIALS 未設定！")
        sys.exit(1)

    # 設置其他參數
    service_name = "speech"
    version = "v1"
    location = "us-central1"  # "us-central1" or "asia-southeast1"
    recognizer_id = "chirp-recognizer"
    language_codes = ["cmn-Hant-TW"]
    model = "chirp_2"

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

    # 列出此 project_id 下所有 recognizers
    list_us_central1_recognizers(
        speech_client=speech_client,
        project_id=PROJECT_ID,
        location=location
    )
