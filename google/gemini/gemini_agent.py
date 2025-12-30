import os
import asyncio

import numpy as np
import librosa
from google import genai
from google.genai import types

from utils import wav_file, to_pcm16_bytes, chunk_bytes


class GeminiAgent:
    """
    Agent for various tasks via Google Gemini API,
    including ASR, translation, dialogue, Q&A, etc.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        live_model_name: str = "gemini-2.5-flash-native-audio-preview-12-2025",
        gemini_api_key: str = "",
        temperature: float = 0.0,
        max_output_tokens: int = 100,
        timeout: int = 5,
        sample_rate: int = 16000,
        chunk_sec: int = 0.1,
        channels: int = 1,
        mime: str = "audio/pcm;rate=16000", # 明確宣告單聲道
    ):
        self.model_name = model_name
        self.live_model_name = live_model_name

        # Get agent configurations
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec
        self.channels = channels
        self.mime = mime

        # Get Gemini API key from config or environment variable
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key is not set. Please provide it in the config "
                "or as an environment variable."
            )

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)

    def process(
        self,
        input_text: str,
        is_live: bool = False,
        response_format: str = "audio"
    ) -> str:
        """使用 Google Gemini API 處理輸入"""
        if is_live:
            return self._asr_live(input_text, response_format=response_format)
        else:
            raise ValueError("ASR mode requires is_live=True.")

    async def _asr_live(self, audio_path: str, response_format: str = "audio", realtime: bool = True):
        """使用 Google Gemini Live 模型 API 進行即時語音轉文字

        realtime:  True=用 sleep 模擬即時上送；False=盡快送完
        """
        config = {
            "response_modalities": [response_format.upper()],
            "input_audio_transcription": {},  # 開啟「輸入音訊」轉寫事件（無可調參）
            "realtime_input_config": {        # 放寬 VAD 的靜默閾值，避免太快結束一個 turn
                "automatic_activity_detection": {
                    "silence_duration_ms": 3000  # 5 秒靜默才視為說完（可依需求調）
                }
            },
        }

        y, _sr = librosa.load(audio_path, sr=self.sample_rate, mono=(self.channels == 1))  # 轉成 16k 單聲道
        pcm_bytes = to_pcm16_bytes(y)
        chunks = chunk_bytes(pcm_bytes, self.chunk_sec, self.sample_rate, self.channels)

        sending_done = asyncio.Event()

        async with self.client.aio.live.connect(
            model=self.live_model_name, config=config
        ) as session:
            print(f"Streaming audio from: {audio_path}")

            async def receiver():
                full = []
                while True:
                    # 每次 receive() 通常只覆蓋「一個 turn」
                    async for msg in session.receive():
                        sc = getattr(msg, "server_content", None)
                        if sc and sc.input_transcription:
                            text = sc.input_transcription.text
                            print(text)
                            full.append(text)

                    # 走到這裡通常是該 turn 完成（VAD 判定結束）
                    if sending_done.is_set():
                        break  # 檔案整體送完了，再結束接收 loop
                return " ".join(full)

            recv_task = asyncio.create_task(receiver())

            for c in chunks:
                await session.send_realtime_input(audio=types.Blob(data=c, mime_type=self.mime))

                if realtime:
                    await asyncio.sleep(self.chunk_sec)

            await session.send_realtime_input(audio_stream_end=True)
            sending_done.set()
            return await recv_task
