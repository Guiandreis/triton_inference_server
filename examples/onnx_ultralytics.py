"""
Ultralytics ONNX Inference Example
=================================

This example runs an exported YOLO ONNX model through the `ultralytics` API.

Notes
-----
- This is an END-TO-END pipeline (Ultralytics preprocessing + ONNX runtime
  execution + Ultralytics postprocessing). It's great for "what users run".
- If you want *engine-only* numbers, benchmark ONNX Runtime directly or use
  Triton's `perf_analyzer`.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from ultralytics import YOLO


class UltralyticsOnnxInference:
    def __init__(
        self,
        onnx_model_path: str,
        imgsz: int = 640,
        device: str = "cpu",
        conf: float = 0.25,
        iou: float = 0.7,
        verbose: bool = False,
    ):
        """
        Initialize an Ultralytics YOLO model backed by an ONNX file.

        Args:
            onnx_model_path: Path to the ONNX model file (e.g. yolov8l.onnx).
            imgsz: Inference image size (square). Commonly 640.
            device: "cpu" or a CUDA device string like "0".
            conf: Confidence threshold for postprocessing.
            iou: IoU threshold for NMS.
            verbose: Whether Ultralytics should be verbose.
        """
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

        self.onnx_model_path = onnx_model_path
        self.imgsz = imgsz
        self.device = device
        self.conf = conf
        self.iou = iou
        self.verbose = verbose

        # Ultralytics auto-selects the backend based on suffix; .onnx uses ORT.
        self.model = YOLO(onnx_model_path)

    def infer(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on a single image (BGR numpy array, as from OpenCV).

        Returns:
            dict with a few common fields for downstream benchmarking.
        """
        # Ultralytics accepts numpy arrays directly; it will handle preprocessing.
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=self.verbose,
        )

        if not results:
            return {"num_detections": 0}

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return {"num_detections": 0}

        # Convert to numpy for easier logging/serialization.
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else None
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else None
        cls = boxes.cls.cpu().numpy() if hasattr(boxes, "cls") else None

        return {
            "num_detections": 0 if xyxy is None else int(xyxy.shape[0]),
            "xyxy": xyxy,
            "conf": conf,
            "cls": cls,
        }


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


if __name__ == "__main__":
    onnx_model_path = os.getenv(
        "ONNX_MODEL_PATH", default="models/yolo11l/yolo11l.onnx"
    )
    input_file = os.getenv("INPUT_FILE", default="examples/test_data/car.jpg")
    imgsz = _env_int("IMGSZ", 640)
    device = os.getenv("DEVICE", default="cpu")
    conf = _env_float("CONF", 0.25)
    iou = _env_float("IOU", 0.7)

    inference_client = UltralyticsOnnxInference(
        onnx_model_path=onnx_model_path,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    if input_file.endswith(".mp4"):
        vidcap = cv2.VideoCapture(input_file)
        while True:
            ret, frame = vidcap.read()
            if not ret:
                break
            out = inference_client.infer(frame)
            print("output", out)
    else:
        image = cv2.imread(input_file)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {input_file}")
        start_time = time.time()
        out = inference_client.infer(image)
        end_time = time.time()
        print("time", end_time - start_time)
        print("output", out)
