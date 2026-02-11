from tritonclient import grpc as grpcclient
import numpy as np
import cv2
from typing import Dict, Any
import os
    
class InferenceClass:

    def __init__(
        self,
        url: str = None,
        model_name: str = "yolov26l"
    ):
        """
        Initialize the inference client.
        
        Args:
            url: Triton server URL (format: host:port). 
            model_name: Name of the model in Triton
        """
        
        
        self.client = grpcclient.InferenceServerClient(url=url, verbose=False)
        self.model_name = model_name
        
        if not self.client.is_server_live():
            raise ConnectionError(f"Triton server is not live at {url}")
        
        if not self.client.is_model_ready(model_name):
            raise ConnectionError(f"Model '{model_name}' is not ready")
            
    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Perform inference on an image.
        
        The image is preprocessed (resize, normalize) before sending to the server.
        
        Args:
            image: Input image as numpy array (RGB format from OpenCV)
            
        Returns:
            dict: Inference result with:
                - output0: numpy array with shape [1, 8400, 84] containing detections
        """
        batch_input = self._preprocess_image(image)
        
        input_tensor = grpcclient.InferInput(
            "images",
            batch_input.shape,
            "FP32"
        )
        input_tensor.set_data_from_numpy(batch_input)
        
        output = grpcclient.InferRequestedOutput("output0")
        
        response = self.client.infer(
            model_name=self.model_name,
            inputs=[input_tensor],
            outputs=[output]
        )
        output_array = response.as_numpy("output0")
        
        return {
            "output0": output_array,
            "shape": output_array.shape
        }

if __name__ == "__main__":
    url = os.getenv('TRITON_URL', default ='localhost:8001')
    model_name = os.getenv("YOLO_MODEL_NAME", default ="yolo11l")
    input_file = os.getenv("INPUT_FILE", default ="test_data/traffic_video.mp4")
    inference_client = InferenceClass(url, model_name)

    if input_file.endswith(".mp4"):
        cap = cv2.VideoCapture(input_file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = inference_client.infer(frame)
            print('output',result)
    else:
        image = cv2.imread(input_file)
        result = inference_client.infer(image)
        print('output',result)