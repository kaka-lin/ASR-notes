import sys

from typing import List, Optional, Tuple
from google.oauth2 import service_account
from googleapiclient.discovery import build, Resource


def get_google_service(
    service_name: str,
    credentials: str,
    version: str = "v3",
    scopes: Optional[List[str]] = None,
) -> Tuple[Resource, service_account.Credentials]:
    """
    通用 Google API client 建立器，回傳 (service, creds)。

    Args:
        service_name (str): 例如 "drive", "speech", "storage"…
        version (str): API 版本，例如 "v3", "v1", "v2"…
        credentials (str): 金鑰檔案路徑
        scopes (Optional[List[str]]): OAuth 範圍清單，預設會用 service_name 自動補

    Returns:
        service (Resource): Google API 服務物件，用於呼叫 API
        creds (service_account.Credentials): 驗證憑證物件，用于 gRPC 或其他需要 Credentials 的 client
    """
    # 若沒有明訂 scopes，就根據 service_name 自動組出 default scope
    if scopes is None:
        base = "https://www.googleapis.com/auth"
        scopes = [f"{base}/{service_name}.readonly",
                  f"{base}/{service_name}"]  # 可按需調整

    try:
        creds = service_account.Credentials.from_service_account_file(
            credentials,
            scopes=scopes
        )
        service = build(service_name, version, credentials=creds)
        return service, creds
    except FileNotFoundError:
        print(f"錯誤：找不到憑證檔案：{credentials}")
        sys.exit(1)
    except ValueError as e:
        print(f"錯誤：無效的憑證或範圍：{e}")
        sys.exit(1)
    except Exception as e:
        print(f"錯誤：建立 {service_name} 服務失敗：{e}")
        sys.exit(1)
