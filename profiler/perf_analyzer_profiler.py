"""
Triton `perf_analyzer` Runner (CPU-serving benchmarks)
=====================================================

This script runs NVIDIA Triton's `perf_analyzer` tool with reasonable defaults
and saves stdout/stderr into a timestamped log directory.

Why this exists
---------------
- `torch.profiler` in this repo is CLIENT-side (Python + gRPC + preprocessing).
- `perf_analyzer` measures SERVER-side performance: throughput, latency, and
  server-side breakdown (queue/compute/overhead), which is what you want for
  Triton CPU comparisons.

Requirements
------------
- `perf_analyzer` must be installed and on PATH.
  - Commonly available inside Triton SDK containers.
  - See Triton docs for installation/usage.

Environment variables
---------------------
- TRITON_URL: default "localhost:8001" (gRPC port)
- YOLO_MODEL_NAME: default "yolo11l"
- PROTOCOL: "grpc" (default) or "http"
- INPUT_NAME: default "images"
- SHAPE: default "1,3,640,640"
- CONCURRENCY_RANGE: default "1:16:1"
- MEASUREMENT_INTERVAL_MS: default "5000"
- INPUT_DATA: default "zero"
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else v


if __name__ == "__main__":
    perf_analyzer = shutil.which("perf_analyzer")
    if perf_analyzer is None:
        raise RuntimeError(
            "perf_analyzer not found on PATH.\n"
            "Install it (often via Triton SDK container) and retry.\n"
            "Docs: https://docs.nvidia.com/deeplearning/triton-inference-server/"
            "user-guide/docs/perf_benchmark/perf_analyzer.html"
        )

    triton_url = _env("TRITON_URL", "localhost:8001")
    model_name = _env("YOLO_MODEL_NAME", "yolo11l")
    protocol = _env("PROTOCOL", "grpc").lower()
    input_name = _env("INPUT_NAME", "images")
    shape = _env("SHAPE", "1,3,640,640")
    concurrency_range = _env("CONCURRENCY_RANGE", "1:16:1")
    measurement_interval_ms = _env("MEASUREMENT_INTERVAL_MS", "5000")
    input_data = _env("INPUT_DATA", "zero")

    if protocol not in {"grpc", "http"}:
        raise ValueError("PROTOCOL must be 'grpc' or 'http'")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("./log") / "perf_analyzer" / model_name / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Compose a conservative perf_analyzer invocation. We capture stdout/stderr
    # to files for blog-ready artifacts.
    cmd = [
        perf_analyzer,
        "-m",
        model_name,
        "-u",
        triton_url,
        "-i",
        protocol,
        "--concurrency-range",
        concurrency_range,
        "--shape",
        f"{input_name}:{shape}",
        "--input-data",
        input_data,
        "--measurement-interval",
        measurement_interval_ms,
        "--percentile",
        "95",
    ]

    (out_dir / "command.txt").write_text(
        " ".join(shlex.quote(c) for c in cmd) + "\n", encoding="utf-8"
    )

    print(f"Running: {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"Logging to: {out_dir}/")

    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    (out_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (out_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    (out_dir / "exit_code.txt").write_text(
        str(completed.returncode) + "\n", encoding="utf-8"
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "perf_analyzer failed.\n"
            f"Exit code: {completed.returncode}\n"
            f"See logs in: {out_dir}/"
        )

    print("perf_analyzer completed successfully.")
    print(f"Output saved under: {out_dir}/")

