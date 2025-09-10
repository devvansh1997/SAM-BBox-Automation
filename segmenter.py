from ultralytics import SAM
import numpy as np

# --- Configuration ---
# As shown in the documentation you provided, we can use the large SAM2 model.
# The library will download the 'sam2_l.pt' file automatically on the first run.
SAM2_MODEL_PATH = "sam2_l.pt" 

class Segmenter:
    def __init__(self):
        """
        Initializes the Segmenter by loading the SAM2 model from the ultralytics package.
        """
        print(f"---- Loading Segmentation Model ----")
        # The same SAM class is used to load both SAM and SAM2 models
        self.model = SAM(SAM2_MODEL_PATH)
        print(f"---- Model {SAM2_MODEL_PATH} has been loaded ----")

    def segment(self, image_path: str, boxes: np.ndarray) -> np.ndarray:
        """
        Generates segmentation masks for a given image using bounding boxes.

        Args:
            image_path (str): The path to the local image file.
            boxes (np.ndarray): A NumPy array of bounding boxes from the detector.
                                Shape: (num_boxes, 4).

        Returns:
            np.ndarray: A NumPy array of boolean masks, where each mask corresponds
                        to a bounding box. Shape: (num_boxes, height, width).
        """
        # 1. Run inference
        # The ultralytics SAM model directly accepts the image path and bounding boxes.
        results = self.model(image_path, bboxes=boxes, verbose=False)

        # 2. Extract the masks
        # If no masks are found, results[0].masks will be None.
        if results[0].masks is None:
            return np.array([]) # Return an empty array if no objects were segmented

        # The masks are stored in the .data attribute as a PyTorch tensor.
        # We move it to the CPU and convert it to a NumPy array for supervision.
        return results[0].masks.data.cpu().numpy()