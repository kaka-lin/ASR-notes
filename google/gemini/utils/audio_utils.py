import numpy as np


def to_pcm16_bytes(y: np.ndarray) -> bytes:
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767.0).astype(np.int16).tobytes()


def chunk_bytes(b: bytes, chunk_sec: int, rate: int = 16000, channels: int = 1) -> list[bytes]:
    bytes_per_sample = 2  # 16-bit
    bytes_per_sec = rate * bytes_per_sample * channels  # ← 乘上聲道數
    bytes_per_chunk = int(bytes_per_sec * chunk_sec)
    return [b[i:i+bytes_per_chunk] for i in range(0, len(b), bytes_per_chunk)]
