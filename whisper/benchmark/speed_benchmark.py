import argparse
import timeit
from functools import partial

from faster_whisper import WhisperModel
from utils import get_logger, inference, get_model

logger = get_logger("speed-benchmark")
parser = argparse.ArgumentParser(description="Speed benchmark")
parser.add_argument(
    "--repeat",
    type=int,
    default=3,
    help="實驗將運行的次數。",
)
parser.add_argument(
    "--number",
    type=int,
    default=10,
    help="在一次重複中函式被呼叫的次數。",
)
parser.add_argument(
    "--model",
    default="large-v3",
    help="要測試的 Whisper 模型名稱。"
)
parser.add_argument(
    "--device-index", type=int, default=0, help="GPU device index"
)
parser.add_argument(
    "--compute_type",
    default="float16",
    choices=["float32", "float16", "int8"],
    help="計算精度。"
)
parser.add_argument(
    "--audio",
    default="benchmark.m4a",
    help="要進行轉錄的音訊檔案路徑。"
)
args = parser.parse_args()
device_idx = args.device_index


def measure_speed():
    logger.info(
        f"正在為模型 '{args.model}' ({args.compute_type}) 測量執行時間"
    )
    logger.info(f"重複次數={args.repeat}, 每次呼叫次數={args.number}")

    # 在計時前載入模型
    print(f"正在初始化模型: {args.model} (精度: {args.compute_type})...")
    model = get_model(args.model, args.compute_type, device_index=device_idx)

    # 建立一個已綁定參數的函式以傳遞給 timeit
    inference_task = partial(inference, model, args.audio)

    # 根據 timeit 文件，應取最小值以獲得真實的計時
    print("開始基準測試...")
    runtimes = timeit.repeat(
        inference_task,
        repeat=args.repeat,
        number=args.number,
    )

    min_runtime = min(runtimes) / args.number

    print("\n--- 速度基準測試結果 ---")
    print(f"模型: {args.model} ({args.compute_type})")
    print(f"音訊: {args.audio}")
    print(f"每次重複的運行時間: {[round(t, 2) for t in runtimes]}")
    print(f"最快執行時間: {min_runtime:.3f}s")


if __name__ == "__main__":
    measure_speed()
