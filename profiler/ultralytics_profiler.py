"""
Ultralytics ONNX Profiler (Torch Profiler style)
================================================

This profiles the *Python-side* execution of Ultralytics' ONNX pipeline using
`torch.profiler` CPU activities (similar to `profiler/torch_profiler.py`).

Important limitation
--------------------
Ultralytics ONNX execution uses ONNX Runtime under the hood. `torch.profiler`
will NOT provide operator-level timing inside ONNX Runtime kernels. It will
still be useful to quantify:
- preprocessing / postprocessing overhead
- Python-side hotspots and allocations
- end-to-end iteration timing breakdown at the Python level

For Triton server-side benchmarking, use `profiler/perf_analyzer_profiler.py`.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import cv2
import torch

# Suppress Python-level warnings (Ultralytics / runtime noise)
warnings.filterwarnings("ignore", category=UserWarning)


def suppress_stderr():
    """Suppress stderr output temporarily (file descriptor redirection)."""

    sys.stderr.flush()
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    return old_stderr, stderr_fd


def restore_stderr(old_stderr, stderr_fd):
    """Restore stderr output."""

    sys.stderr.flush()
    os.dup2(old_stderr, stderr_fd)
    os.close(old_stderr)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


if __name__ == "__main__":
    # Allow imports from project root.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    onnx_model_path = os.getenv(
        "ONNX_MODEL_PATH", default="models/yolo11l/yolo11l.onnx"
    )
    input_file = os.getenv("INPUT_FILE", default="examples/test_data/car.jpg")
    imgsz = _env_int("IMGSZ", 640)
    device = os.getenv("DEVICE", default="cpu")
    conf = _env_float("CONF", 0.25)
    iou = _env_float("IOU", 0.7)

    wait = _env_int("PROFILE_WAIT", 1)
    warmup = _env_int("PROFILE_WARMUP", 1)
    active = _env_int("PROFILE_ACTIVE", 10)
    extra_iters = _env_int("PROFILE_EXTRA_ITERS", 3)
    num_iterations = wait + warmup + active + extra_iters

    from examples.onnx_ultralytics import UltralyticsOnnxInference

    image = cv2.imread(input_file)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {input_file}")

    model_stem = Path(onnx_model_path).stem
    log_dir = Path("./log") / "ultralytics" / model_stem
    log_dir.mkdir(parents=True, exist_ok=True)

    # Model init can trigger backend checks / logs; optionally suppress.
    old_stderr, stderr_fd = suppress_stderr()
    try:
        inference_client = UltralyticsOnnxInference(
            onnx_model_path=onnx_model_path,
            imgsz=imgsz,
            device=device,
            conf=conf,
            iou=iou,
            verbose=False,
        )
    finally:
        restore_stderr(old_stderr, stderr_fd)

    old_stderr, stderr_fd = suppress_stderr()
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(log_dir)),
            profile_memory=True,
            with_stack=True,
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
        ) as prof:
            for _ in range(num_iterations):
                _ = inference_client.infer(image)
                prof.step()
    finally:
        restore_stderr(old_stderr, stderr_fd)

    print(f"\nProfiler completed {num_iterations} iterations")
    print(f"Logs saved to: {log_dir}/")
    print("\nTop CPU operations:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

