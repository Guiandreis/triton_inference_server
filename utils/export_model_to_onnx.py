from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model_onnx = model.export(format="onnx", imgsz=640, dynamic=True, opset=11, simplify=False, optimize=True, half=True, device = "cpu", nms = True )
