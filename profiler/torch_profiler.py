"""
Triton Client-Side Profiler
============================
This profiler captures CLIENT-SIDE operations (gRPC calls, preprocessing, etc.)
NOT the actual model inference operations inside Triton server.

For operator-level profiling of the model itself, use torch_model_profiler.py
"""
import torch
import sys
import os
import cv2
import warnings

# Suppress Python-level warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Suppress C++ level CUDA errors by redirecting stderr at file descriptor level
def suppress_stderr():
    """Suppress stderr output temporarily"""
    sys.stderr.flush()
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    os.close(devnull)
    return old_stderr, stderr_fd

def restore_stderr(old_stderr, stderr_fd):
    """Restore stderr output"""
    sys.stderr.flush()
    os.dup2(old_stderr, stderr_fd)
    os.close(old_stderr)

# Add project root to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Temporarily suppress stderr during CUDA initialization
old_stderr, stderr_fd = suppress_stderr()
from examples.onnx_triton import InferenceClass
restore_stderr(old_stderr, stderr_fd)

model_name = "yolov8l"
url = "localhost:8001"
input_file = "examples/test_data/car.jpg"

# Suppress stderr during client initialization (may trigger CUDA checks)
old_stderr, stderr_fd = suppress_stderr()
inference_client = InferenceClass(url, model_name)
restore_stderr(old_stderr, stderr_fd)

image = cv2.imread(input_file)

# Create log directory if it doesn't exist
os.makedirs(f'./log/{model_name}', exist_ok=True)

# Number of iterations to profile
# wait=1 (skip 1), warmup=1 (warmup 1), active=10 (profile 10)
# Total iterations needed: 1 + 1 + 10 = 12
num_iterations = 15  # Run a few extra to ensure profiler completes

# Suppress stderr during profiler setup and execution
old_stderr, stderr_fd = suppress_stderr()
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    record_shapes=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model_name}'),
    profile_memory=True,
    with_stack=True,
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=10)
) as prof:
    for i in range(num_iterations):
        result = inference_client.infer(image)
        prof.step()
restore_stderr(old_stderr, stderr_fd)

print(f"\nProfiler completed {num_iterations} iterations")
print(f"Logs saved to: ./log/{model_name}/")
print("\nTop CPU operations:")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))