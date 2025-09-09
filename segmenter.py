import torch
from PIL import Image
from transformers import SamModel, SamProcessor
import numpy as np
import config

class Segmenter:
    def __init__(self):
        print("---- Loading Segmentation Model ----")
        # load model on GPU
        self.device = torch.device(config.DEVICE)
        self.model = SamModel.from_pretrained(config.SEG_MODEL_ID).to(self.device)
        self.processor = SamProcessor.from_pretrained(config.SEG_MODEL_ID)
        print(f"---- Model {config.SEG_MODEL_ID} has been loaded on {config.DEVICE} ----")
    
    def segment(self, image_path: str, boxes: np.ndarray) -> np.ndarray:
        """
        Generates segmentation masks for a given image using bounding boxes.

        Args:
            image_path (str): The path to the local image file.
            boxes (np.ndarray): A NumPy array of bounding boxes from Grounding DINO.
                                Shape: (num_boxes, 4).

        Returns:
            np.ndarray: A NumPy array of boolean masks, where each mask corresponds
                        to a bounding box. Shape: (num_boxes, height, width).
        """
        # load the image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size[::-1]

        # prepare image and bounding box for model
        # sam2 wants nested list bbox
        input_boxes = [[box.tolist() for box in boxes]]
        inputs = self.processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(self.device)
        
        # get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        all_masks_tensor = self.processor.post_process_masks(
            outputs.pred_masks,
            original_sizes=[original_size],
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0] # Shape: (num_boxes, 3, H, W)

        iou_scores = outputs.iou_scores.squeeze(0)

        best_mask_indices = torch.argmax(iou_scores, dim=1)

        final_masks = []
        for i, idx in enumerate(best_mask_indices):
            # Select the mask at the best index (idx) for the i-th detection
            best_mask = all_masks_tensor[i, idx, :, :]
            final_masks.append(best_mask)

        if not final_masks:
            return np.array([]) # Return empty array if no masks
            
        final_masks_tensor = torch.stack(final_masks, dim=0)
        
        return final_masks_tensor.cpu().numpy()