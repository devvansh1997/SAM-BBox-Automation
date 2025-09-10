# In yolo_detector.py

from ultralytics import YOLO
import supervision as sv
import numpy as np

class YOLODetector:
    def __init__(self, model_path: str = 'yolo11x.pt'):
        """
        Initializes the YOLO detector with a pre-trained model.

        Args:
            model_path (str): The path to the YOLOv8 model file (e.g., 'yolov8n.pt').
        """
        print("---- Loading YOLOv8 Model ----")
        self.model = YOLO(model_path)
        # The model.names attribute is a dictionary mapping class IDs to class names.
        # e.g., {0: 'person', 1: 'bicycle', 2: 'car', ...}
        self.class_names = self.model.names
        print(f"---- Model {model_path} has been loaded ----")

        # Define the target classes we want to detect
        self.target_classes = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']
        # Get the corresponding class IDs for our target classes
        self.target_class_ids = [
            k for k, v in self.class_names.items() if v in self.target_classes
        ]

    def predict(self, image_path: str, conf_threshold: float = 0.25) -> sv.Detections:
        """
        Performs inference and filters for target classes.

        Args:
            image_path (str): Path to the input image.
            conf_threshold (float): The confidence threshold for YOLO detections.

        Returns:
            sv.Detections: A supervision Detections object containing only the
                           bounding boxes for the target classes.
        """
        # 1. Perform detection
        results = self.model(image_path, conf=conf_threshold, verbose=False)[0]
        
        # 2. Convert the results to a supervision.Detections object
        detections = sv.Detections.from_ultralytics(results)

        # 3. Create a boolean mask for our target classes
        # np.isin() checks for each detection if its class_id is in our target list
        mask = np.isin(detections.class_id, self.target_class_ids)

        # 4. Apply the mask to filter the detections
        filtered_detections = detections[mask]

        return filtered_detections