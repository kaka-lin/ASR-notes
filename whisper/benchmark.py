import os
import time
import threading
import traceback
from queue import Queue
import subprocess
from collections import defaultdict
import statistics

import torch
from faster_whisper import WhisperModel

# test audio file path
AUDIO_PATH = "test_audio.wav"

# 請求數量
REQUESTS = 4

# model config
MODEL_PATH = "Systran/faster-whisper-large-v3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# 擷取邏輯核心數
def get_cpu_count():
    c = os.cpu_count()
    return c if c is not None else 1

# CPU 核心數與每請求核心數
CPU_CORES = get_cpu_count()
PER_REQUEST_CORES = max(1, CPU_CORES // REQUESTS)

# 待選的 cpu_threads 和 num_workers
CPU_THREADS_LIST = [2**i for i in range(CPU_CORES.bit_length()) if 2**i < CPU_CORES]
NUM_WORKER_LIST = [2**i for i in range(CPU_CORES.bit_length()) if 2**i < CPU_CORES]

# 全局鎖 (Singleton 模式下保護單一模型)
_singleton_lock = threading.Lock()


def get_gpu_memory():
    """監測 GPU 記憶體使用量 (GB)"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        memory_used = result.stdout.strip().split("\n")  # 取得所有 GPU 的記憶體使用狀態
        memory_used = [float(mem) / 1024 for mem in memory_used]  # 轉換成 GB
        return memory_used[0] if len(memory_used) > 0 else 0.0
    except Exception as e:
        print(f"⚠️ 無法獲取 GPU 記憶體使用量: {e}")
        return 0.0


def print_top_k_and_lowest_resource(summary, label, k=3):
    print(f"\n=== {label} - Top {k} 最佳平均延遲組合 ===")
    sorted_summary = sorted(summary, key=lambda x: x["mean_time"])
    top_k = sorted_summary[:k]
    for entry in top_k:
        print(f" [{label}][cpu_threads={entry['cpu_threads']}, num_workers={entry['num_workers']}] "
              f"耗時 {entry['mean_time']:.3f}s (StdDev: {entry['std_time']:.3f})")

    # 資源消耗最小的組合 (cpu_threads + num_workers 最小，且耗時與最低耗時差 < 5%)
    min_time = top_k[0]["mean_time"]
    best_efficiency = min(
        (e for e in top_k if e["mean_time"] <= min_time * 1.05),  # 差異小於5%
        key=lambda e: e["cpu_threads"] + e["num_workers"]
    )
    print(f"\n👉 建議使用低資源組合："
          f"[{label}][cpu_threads={best_efficiency['cpu_threads']}, num_workers={best_efficiency['num_workers']}] "
          f"耗時 {best_efficiency['mean_time']:.3f}s (StdDev: {best_efficiency['std_time']:.3f})")


def download_model(model_path, download_root="whisper_models"):
    """下載模型到指定目錄"""
    try:
        model = WhisperModel(
            model_path,
            download_root=download_root,
            local_files_only=False,  # 允許從 HuggingFace 下載
        )
        print(f"✅ 模型 {model_path} 下載成功")
    except Exception as e:
        print(f"❌ 模型 {model_path} 下載失敗: {e}")


def single_request_benchmark(cpu_threads, num_workers, download_root="whisper_models"):
    """單請求測試，返回總耗時"""
    try:
        model = WhisperModel(
            MODEL_PATH,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
            download_root=download_root,
            local_files_only=True,  # 確保使用本地模型
        )
        start = time.time()
        model.transcribe(AUDIO_PATH)
        elapsed = time.time() - start
        return elapsed
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"    [單請求][cpu_threads={cpu_threads},num_workers={num_workers}] 失敗: CUDA OOM")
        else:
            print(f"    [單請求][cpu_threads={cpu_threads},num_workers={num_workers}] 失敗: RuntimeError - {str(e)}")
        return None
    except Exception as e:
        print(f"    [單請求][cpu_threads={cpu_threads},num_workers={num_workers}] 失敗: {e.__class__.__name__}")
        return None



#############################################
# 方案1 (Singleton): 所有 thread 共享一個 model 實例
#############################################
def test_shared_model(cpu_threads, num_workers, download_root="whisper_models"):
    times = []
    lock = threading.Lock()
    oom_flag = False

    # 建立共用的 model 實例
    shared_model = WhisperModel(
        MODEL_PATH,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        download_root=download_root,
        local_files_only=True,  # 確保使用本地模型
    )

    def worker(room_id):
        nonlocal oom_flag
        try:
            start = time.time()
            with _singleton_lock:
                # 確保只有一個 thread 在使用 shared_model
                shared_model.transcribe(AUDIO_PATH)
            elapsed = time.time() - start
            with lock:
                times.append(elapsed)
            print(f"    [請求 {room_id}][Singleton][cpu_threads={cpu_threads},num_workers={num_workers}] 耗時 {elapsed:.3f}s")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_flag = True
                print(f"    [請求 {room_id}][Singleton][cpu_threads={cpu_threads},num_workers={num_workers}] 失敗: CUDA OOM")
            else:
                print(f"    [請求 {room_id}][Singleton][cpu_threads={cpu_threads},num_workers={num_workers}] RuntimeError: {e}")
        except Exception as e:
            oom_flag = True
            print(f"    [請求 {room_id}][Singleton][cpu_threads={cpu_threads},num_workers={num_workers}] 失敗: {e.__class__.__name__}")

    threads = []
    for i in range(REQUESTS):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if oom_flag or len(times) < REQUESTS:
        return None, None
    total_time = sum(times)
    avg_time = total_time / len(times) if times else 0
    return total_time, avg_time


#############################################
# 方案2 (Non-Singleton)：每個 thread 獨立創建 model 實例
#############################################
def test_individual_model(cpu_threads, num_workers, download_root="whisper_models"):
    times = []
    lock = threading.Lock()
    oom_flag = False

    def worker(room_id):
        nonlocal oom_flag
        try:
            # 每個 thread 單獨創建 model 實例
            model = WhisperModel(
                MODEL_PATH,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
                download_root=download_root,
                local_files_only=True,  # 確保使用本地模型
            )

            start = time.time()
            model.transcribe(AUDIO_PATH)
            elapsed = time.time() - start
            with lock:
                times.append(elapsed)
            print(f"    [請求 {room_id}][Non‐Singleton][cpu_threads={cpu_threads},num_workers={num_workers}] 耗時 {elapsed:.3f}s")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_flag = True
                print(f"    [請求 {room_id}][Non‐Singleton][cpu_threads={cpu_threads},num_workers={num_workers}] 失敗: CUDA OOM")
            else:
                print(f"    [請求 {room_id}][Non‐Singleton][cpu_threads={cpu_threads},num_workers={num_workers}] RuntimeError: {e}")
        except Exception as e:
            oom_flag = True
            print(f"    [請求 {room_id}][Non‐Singleton][cpu_threads={cpu_threads},num_workers={num_workers}] 失敗: {e.__class__.__name__}")

    threads = []
    for i in range(REQUESTS):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if oom_flag or len(times) < REQUESTS:
        return None, None
    total_time = sum(times)
    avg_time = total_time / len(times) if times else 0
    return total_time, avg_time


#############################################
# 執行不同參數組合測試
#############################################
def run_all_benchmarks(repetitions=1):
    if not os.path.isfile(AUDIO_PATH):
        print(f"❌ 找不到音訊檔：{AUDIO_PATH}")
        return None, None

    # dowmload 模型 first
    print(f"正在下載模型：{MODEL_PATH} ...")
    download_model(MODEL_PATH)

    print(f"\n=== 開始全面 Benchmark ===")
    print(f"音訊檔：{AUDIO_PATH}")
    print(f"模型大小：{MODEL_PATH}")
    print(f"請求數量：{REQUESTS}，CPU 總核心：{CPU_CORES}，平均每請求 cores：{PER_REQUEST_CORES}\n")

    # 選擇要嘗試的 cpu_threads / num_workers 組合
    # 因為 cpu_thread * num_workers < CPU_CORES
    # 所以我們只選擇 <= PER_REQUEST_CORES 的組合
    cpu_threads_list = [t for t in CPU_THREADS_LIST if t <= PER_REQUEST_CORES]
    num_worker_list = [n for n in NUM_WORKER_LIST if n <= PER_REQUEST_CORES]
    print(f"可用的 cpu_threads: {cpu_threads_list}")
    print(f"可用的 num_workers: {num_worker_list}")

    # 儲存多次測試結果
    all_single_results = []  # 會有 repeats * len(cpu_threads_list)*len(num_worker_list) 筆資料
    all_multi_results  = []

    for rep in range(repetitions):
        print(f"\n--- 第 {rep+1} 次重複測試開始 ---\n")

        # 1. 先做單請求測試，確認模型是否能正常運行
        print(f"=== [單請求測試] ===")
        for cpu_threads in cpu_threads_list:
            for num_workers in num_worker_list:
                elapsed = single_request_benchmark(cpu_threads, num_workers)
                if elapsed is not None:
                    print(f"  [單請求][cpu_threads={cpu_threads}, num_workers={num_workers}] 耗時 {elapsed:.3f}s")
                else:
                    print(f"  [單請求][cpu_threads={cpu_threads}, num_workers={num_workers}] OOM 或失敗")
                # 儲存結果
                all_single_results.append({
                    "mode": "single-room",
                    "cpu_threads": cpu_threads,
                    "num_workers": num_workers,
                    "avg_time": elapsed,
                    "oom": (elapsed is None)
                })

        # 2. 根據單請求測試結果，進行多請求測試: Singleton 模式
        print(f"=== [多請求測試] Singleton ===")
        for cpu_threads in cpu_threads_list:
            for num_workers in num_worker_list:
                total_time_s, avg_time_s = test_shared_model(cpu_threads, num_workers)
                if avg_time_s is not None:
                    print(f"  [Singleton][cpu_threads={cpu_threads}, num_workers={num_workers}] 總耗時={total_time_s:.3f}s, 平均={avg_time_s:.3f}s")
                else:
                    print(f"  [Singleton][cpu_threads={cpu_threads}, num_workers={num_workers}] 失敗 (OOM 或 其他錯誤)")
                # 儲存結果
                all_multi_results.append({
                    "mode": "singleton",
                    "cpu_threads": cpu_threads,
                    "num_workers": num_workers,
                    "total_time": total_time_s,
                    "avg_time": avg_time_s,
                    "oom": (avg_time_s is None)
                })

        # 3. 根據單請求測試結果，進行多請求測試: Non-Singleton 模式
        print(f"=== [多請求測試] Non-Singleton ===")
        for cpu_threads in cpu_threads_list:
            for num_workers in num_worker_list:
                total_time_n, avg_time_n = test_individual_model(cpu_threads, num_workers)
                if avg_time_n is not None:
                    print(f"  [Non‐Singleton][cpu_threads={cpu_threads}, num_workers={num_workers}] 總耗時={total_time_n:.3f}s, 平均={avg_time_n:.3f}s")
                else:
                    print(f"  [Non‐Singleton][cpu_threads={cpu_threads}, num_workers={num_workers}] 失敗 (OOM 或 其他錯誤)")
                # 儲存結果
                all_multi_results.append({
                    "mode": "non-singleton",
                    "cpu_threads": cpu_threads,
                    "num_workers": num_workers,
                    "total_time": total_time_n,
                    "avg_time": avg_time_n,
                    "oom": (avg_time_n is None)
                })

        print(f"--- 第 {rep+1} 次重複測試結束 ---\n")

    # === 統計「單請求最佳」 ===
    #   先篩掉 OOM，取出每種 (cpu_threads, num_workers) 在 repetitions 次裡面的 avg_time 平均值
    single_aggregate = defaultdict(list)  # key=(cpu_threads, num_workers) -> [list of avg_time]
    for r in all_single_results:
        key = (r["cpu_threads"], r["num_workers"])
        if not r["oom"] and r["avg_time"] is not None:
            single_aggregate[key].append(r["avg_time"])

    single_summary = []  # 會放 {"cpu_threads":..., "num_workers":..., "mean_time":..., "std":...}
    for key, times in single_aggregate.items():
        cpu_threads, num_workers = key
        mean_t = statistics.mean(times)
        std_t = statistics.pstdev(times) if len(times) > 1 else 0.0
        single_summary.append({
            "cpu_threads": cpu_threads,
            "num_workers": num_workers,
            "mean_time": mean_t,
            "std_time": std_t
        })

    if single_summary:
        best_single = min(single_summary, key=lambda x: x["mean_time"])
        print("=== 單請求 - 重複測試後最佳配置 (Single-Request Best) ===")
        print(f" cpu_threads : {best_single['cpu_threads']}")
        print(f" num_workers : {best_single['num_workers']}")
        print(f" 平均延遲    : {best_single['mean_time']:.3f}s  (StdDev: {best_single['std_time']:.3f})")

        print_top_k_and_lowest_resource(single_summary, label="單請求", k=3)
    else:
        print("❌ 單請求所有配置皆 OOM 或失敗。")

    # === 統計「多請求併發最佳」 ===
    multi_aggregate = defaultdict(list)  # key=(mode, cpu_threads, num_workers)
    for r in all_multi_results:
        key = (r["mode"], r["cpu_threads"], r["num_workers"])
        if not r["oom"] and r["avg_time"] is not None:
            multi_aggregate[key].append(r["avg_time"])

    multi_summary = []
    for key, times in multi_aggregate.items():
        mode, cpu_threads, num_workers = key
        mean_t = statistics.mean(times)
        std_t = statistics.pstdev(times) if len(times) > 1 else 0.0
        multi_summary.append({
            "mode": mode,
            "cpu_threads": cpu_threads,
            "num_workers": num_workers,
            "mean_time": mean_t,
            "std_time": std_t
        })

    if multi_summary:
        best_multi = min(multi_summary, key=lambda x: x["mean_time"])
        print(f"\n=== {REQUESTS}請求併發 - 重複測試後最佳配置 ({REQUESTS}-Requests Concurrency Best) ===")
        print(f" 模式        : {best_multi['mode']}")
        print(f" cpu_threads : {best_multi['cpu_threads']}")
        print(f" num_workers : {best_multi['num_workers']}")
        print(f" 平均延遲    : {best_multi['mean_time']:.3f} s  (StdDev: {best_multi['std_time']:.3f})")

        # 過濾相同 mode
        for mode in sorted(set(r["mode"] for r in multi_summary)):
            mode_summary = [r for r in multi_summary if r["mode"] == mode]
            print_top_k_and_lowest_resource(mode_summary, label=f"{REQUESTS}請求-{mode}", k=3)
    else:
        print(f"❌ {REQUESTS}請求併發ㄧ所有配置皆 OOM 或失敗。")

    return best_single, best_multi  # 回傳給外層呼叫者


def main():
    repetitions = 1
    run_all_benchmarks(repetitions=repetitions)


if __name__ == "__main__":
    main()
