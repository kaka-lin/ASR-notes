import os
import math
import requests
import subprocess

from dotenv import load_dotenv
from moviepy import VideoFileClip

# 載入 .env 內容到環境變數，並強制更新
if not load_dotenv(override=True):
    print("警告：.env 檔案不存在或解析失敗，請確認它位於專案根目錄。")


def download_direct_video(
        url: str,
        output_path: str = ".",
        filename: str = "test_video.mp4") -> str:
    """
    從指定 URL 下載影片並儲存為指定檔案。

    Args:
        url (str): YouTube 影片網址。
        output_path (str): 影片儲存資料夾。
        filename (str): 下載後的檔名（含副檔名 .mp4）。

    Returns:
        str: 下載後影片的完整路徑。
    """
    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, filename)

    print(f"📥 下載影片：{url}")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"✅ 影片下載完成：{file_path}")
        return file_path
    else:
        print(f"❌ 下載失敗，HTTP 狀態碼：{response.status_code}")
        exit(1)


def download_youtube_video(
        url: str,
        output_path: str = ".",
        filename: str = "test_video.mp4") -> str:
    """
    下載 YouTube 影片並儲存為 mp4。

    Args:
        url (str): YouTube 影片網址。
        output_path (str): 影片儲存資料夾。
        filename (str): 下載後的檔名（含副檔名 .mp4）。

    Returns:
        str: 下載後影片的完整路徑。
    """
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, filename)

    print(f"📥 下載影片：{url}")
    cmd = [
        "yt-dlp",
        "-f", "mp4",
        "-o", filepath,
        url
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ 影片下載完成：{filepath}")

    return filepath


def extract_audio(video_path: str, output_dir: str, segment_duration: int = 30):
    """
    從影片檔案中提取音訊，並將其轉為 16kHz mono 的 .wav 格式。
    若指定 segment_duration，則會自動切割為多個檔案。

    Args:
        video_path (str): 輸入的 mp4 影片路徑。
        output_dir (str): 輸出的音訊資料夾。
        segment_duration (int): 每段音訊的長度（秒）。預設為 30 秒。
    """
    print(f"🎵 正在提取音訊：{video_path}")
    os.makedirs(output_dir, exist_ok=True)

    video = VideoFileClip(video_path)
    if video.audio is None:
        print("❌ 影片中未找到音訊！")
        return

    audio = video.audio
    duration = audio.duration
    num_segments = math.ceil(duration / segment_duration)

    print(f"📌 總音訊時長：{duration:.2f} 秒，切割為 {num_segments} 段，每段 {segment_duration} 秒")

    for i in range(num_segments):
        start = i * segment_duration
        end = min((i + 1) * segment_duration, duration) # 確保不超出範圍
        output_filename = os.path.join(output_dir, f"audio_part_{i+1:02d}.wav")

        print(f"🔹 處理時間段：{start:.2f} ~ {end:.2f} 秒 -> {output_filename}")

        # 擷取該時間範圍的音訊
        # **使用 `subclipped()`（MoviePy 2.1.2 版本）**
        video_segment = video.subclipped(start, end)
        segment = video_segment.audio  # 取得剪輯後的音訊
        segment.write_audiofile(
            output_filename,
            codec="pcm_s16le",
            fps=16000,
            ffmpeg_params=["-ac", "1"] # 確保輸出為單聲道
        )

    print(f"✅ 音訊切割完成，儲存至 {output_dir}")


if __name__ == "__main__":
    # 設定下載網址 & 檔案名稱
    VIDEO_URL = "https://www.youtube.com/watch?v=fBbaxlIEppE"  # 替換為實際的 YouTube 影片網址
    VIDEO_NAME = "test_video.mp4"
    OUTPUT_DIR = os.getenv("DOWNLOAD_DIR", "downloads")
    VIDEO_PATH = os.path.join(OUTPUT_DIR, VIDEO_NAME)
    AUDIO_DIR = os.path.join(OUTPUT_DIR, "audios")

    # 下載影片
    if not os.path.exists(VIDEO_PATH):
        download_youtube_video(
            url=VIDEO_URL,
            output_path=OUTPUT_DIR,
            filename=VIDEO_NAME)

    # Extract audio
    extract_audio(
        video_path=VIDEO_PATH,
        output_dir=AUDIO_DIR,
        segment_duration=30  # 每段 30 秒
    )
