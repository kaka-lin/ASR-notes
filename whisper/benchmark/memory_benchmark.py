import argparse
import time
from typing import Callable

import py3nvml.py3nvml as nvml
import torch
from memory_profiler import memory_usage

from utils import MyThread, get_logger, inference, get_model

logger = get_logger("memory-benchmark")
parser = argparse.ArgumentParser(description="Memory benchmark")
parser.add_argument(
    "--gpu_memory", action="store_true", help="Measure GPU memory usage"
)
parser.add_argument(
    "--interval",
    type=float,
    default=0.1,
    help="Interval at which measurements are collected",
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
interval = args.interval


def measure_memory(func: Callable):
    model = None
    if args.gpu_memory:
        logger.info(
            "Measuring maximum GPU memory usage on GPU device."
            " Make sure to not have additional processes running on the same GPU."
        )

        # init nvml
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(device_idx)
        gpu_name = nvml.nvmlDeviceGetName(handle)
        gpu_memory_limit = nvml.nvmlDeviceGetMemoryInfo(handle).total >> 20
        gpu_power_limit = nvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
        info = {"gpu_memory_usage": [], "gpu_power_usage": []}

        # 確保 VRAM 是乾淨的
        torch.cuda.empty_cache()
        # 記錄監控前的 VRAM
        used_before = nvml.nvmlDeviceGetMemoryInfo(handle).used >> 20

        def _monitor_gpu():
            """Background thread to monitor GPU stats using closure."""
            nonlocal stop_monitoring
            while True:
                info["gpu_memory_usage"].append(
                    nvml.nvmlDeviceGetMemoryInfo(handle).used >> 20
                )
                info["gpu_power_usage"].append(
                    nvml.nvmlDeviceGetPowerUsage(handle) / 1000
                )
                time.sleep(interval)

                if stop_monitoring:
                    break
            return info

        stop_monitoring = False
        thread = MyThread(_monitor_gpu, params=())
        thread.start()

        # run the function to measure memory usage
        print(f"正在初始化模型: {args.model} (精度: {args.compute_type})...")
        model = get_model(args.model, args.compute_type, device_index=device_idx)
        func(model, args.audio)

        stop_monitoring = True
        thread.join()
        result = thread.get_result()

        # shutdown nvml
        nvml.nvmlShutdown()

        if not result["gpu_memory_usage"]:
            logger.error("Failed to collect GPU memory usage data.")
            return None

        max_memory_usage_absolute = max(result["gpu_memory_usage"])
        max_memory_usage_delta = max_memory_usage_absolute - used_before
        max_power_usage = max(result["gpu_power_usage"])

        print("\n--- Memory Benchmark Results ---")
        print("GPU name: %s" % gpu_name)
        print("GPU device index: %s" % device_idx)
        print(f"VRAM usage before load: {used_before} MiB")
        print(f"Peak absolute VRAM usage: {max_memory_usage_absolute} MiB")
        print(
            "Maximum GPU memory usage (Delta): %dMiB / %dMiB (%.2f%%)"
            % (
                max_memory_usage_delta,
                gpu_memory_limit,
                (max_memory_usage_delta / gpu_memory_limit) * 100,
            )
        )
        print(
            "Maximum GPU power usage: %dW / %dW (%.2f%%)"
            % (
                max_power_usage,
                gpu_power_limit,
                (max_power_usage / gpu_power_limit) * 100,
            )
        )
    else:
        logger.info("Measuring maximum increase of memory usage.")
        print(f"正在初始化模型: {args.model} (精度: {args.compute_type})...")
        model = get_model(args.model, args.compute_type, device_index=device_idx)
        max_usage = memory_usage(
            (func, (model, args.audio)),
            max_usage=True,
            interval=interval,
        )
        print("Maximum increase of RAM memory usage: %d MiB" % max_usage)

    return model


if __name__ == "__main__":
    # measure memory usage during inference
    model = measure_memory(inference)

    # delete the model to free memory
    if model:
        print("Measurement finished. Deleting model to free memory...")
        del model

    # 搭配 torch.cuda.empty_cache() 可以強制將快取的 GPU 記憶體還給系統
    if args.gpu_memory:
        torch.cuda.empty_cache()
