from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import psutil
import torch


# ---------------------------------------------------------
# Benchmark Result
# ---------------------------------------------------------

@dataclass
class BenchmarkResult:
    method: str

    total_samples: int

    total_time_seconds: float

    average_latency_seconds: float

    min_latency_seconds: float

    max_latency_seconds: float

    throughput_samples_per_second: float

    average_input_tokens: float

    average_output_tokens: float

    peak_memory_mb: float

    gpu_memory_mb: float


# ---------------------------------------------------------
# Token Utilities
# ---------------------------------------------------------

def estimate_token_count(
    text: str,
) -> int:
    """
    Rough token estimation.

    Useful for benchmarking
    without tokenizer dependency.
    """

    return max(
        1,
        len(text.split()),
    )


# ---------------------------------------------------------
# Memory Utilities
# ---------------------------------------------------------

def get_process_memory_mb() -> float:
    """
    Current RAM usage.
    """

    process = psutil.Process()

    memory_bytes = (
        process.memory_info().rss
    )

    return memory_bytes / (1024 ** 2)


def get_gpu_memory_mb() -> float:
    """
    Current GPU memory usage.
    """

    if not torch.cuda.is_available():
        return 0.0

    memory_bytes = (
        torch.cuda.max_memory_allocated()
    )

    return memory_bytes / (1024 ** 2)


# ---------------------------------------------------------
# Main Benchmark
# ---------------------------------------------------------

def benchmark_summarizer(
    texts: List[str],
    summarize_fn: Callable[[str], str],
    method_name: str,
) -> BenchmarkResult:
    """
    Benchmark summarization pipeline.

    Measures:
    -----------------------------------------
    - latency
    - throughput
    - RAM usage
    - GPU memory
    - token statistics
    -----------------------------------------
    """

    if not texts:

        raise ValueError(
            "No input texts provided."
        )

    print(
        f"[BENCHMARK] Running benchmark "
        f"for: {method_name}"
    )

    latencies = []

    input_tokens = []

    output_tokens = []

    peak_memory = get_process_memory_mb()

    # ---------------------------------------------------------
    # GPU reset
    # ---------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats()

    # ---------------------------------------------------------
    # Benchmark loop
    # ---------------------------------------------------------

    start_total = time.perf_counter()

    for idx, text in enumerate(texts):

        input_tokens.append(
            estimate_token_count(text)
        )

        start = time.perf_counter()

        summary = summarize_fn(text)

        end = time.perf_counter()

        latency = end - start

        latencies.append(latency)

        output_tokens.append(
            estimate_token_count(summary)
        )

        current_memory = (
            get_process_memory_mb()
        )

        peak_memory = max(
            peak_memory,
            current_memory,
        )

        print(
            f"[BENCHMARK] "
            f"{idx + 1}/{len(texts)} "
            f"| latency={latency:.2f}s"
        )

    total_time = (
        time.perf_counter()
        - start_total
    )

    # ---------------------------------------------------------
    # GPU memory
    # ---------------------------------------------------------

    gpu_memory = get_gpu_memory_mb()

    # ---------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------

    avg_latency = statistics.mean(
        latencies
    )

    throughput = (
        len(texts) / total_time
    )

    result = BenchmarkResult(
        method=method_name,

        total_samples=len(texts),

        total_time_seconds=round(
            total_time,
            4,
        ),

        average_latency_seconds=round(
            avg_latency,
            4,
        ),

        min_latency_seconds=round(
            min(latencies),
            4,
        ),

        max_latency_seconds=round(
            max(latencies),
            4,
        ),

        throughput_samples_per_second=round(
            throughput,
            4,
        ),

        average_input_tokens=round(
            float(np.mean(input_tokens)),
            2,
        ),

        average_output_tokens=round(
            float(np.mean(output_tokens)),
            2,
        ),

        peak_memory_mb=round(
            peak_memory,
            2,
        ),

        gpu_memory_mb=round(
            gpu_memory,
            2,
        ),
    )

    return result


# ---------------------------------------------------------
# Pretty Printing
# ---------------------------------------------------------

def print_benchmark_result(
    result: BenchmarkResult,
) -> None:
    """
    Pretty-print benchmark results.
    """

    print("\n" + "=" * 60)

    print(
        f"BENCHMARK RESULTS "
        f"[{result.method}]"
    )

    print("=" * 60)

    print(
        f"Samples Processed: "
        f"{result.total_samples}"
    )

    print(
        f"Total Time: "
        f"{result.total_time_seconds:.2f}s"
    )

    print(
        f"Average Latency: "
        f"{result.average_latency_seconds:.2f}s"
    )

    print(
        f"Min Latency: "
        f"{result.min_latency_seconds:.2f}s"
    )

    print(
        f"Max Latency: "
        f"{result.max_latency_seconds:.2f}s"
    )

    print(
        f"Throughput: "
        f"{result.throughput_samples_per_second:.2f} "
        f"samples/sec"
    )

    print(
        f"Average Input Tokens: "
        f"{result.average_input_tokens}"
    )

    print(
        f"Average Output Tokens: "
        f"{result.average_output_tokens}"
    )

    print(
        f"Peak RAM Usage: "
        f"{result.peak_memory_mb:.2f} MB"
    )

    print(
        f"Peak GPU Memory: "
        f"{result.gpu_memory_mb:.2f} MB"
    )

    print("=" * 60)