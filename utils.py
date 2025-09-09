# In utils.py

import cv2
import supervision as sv
from PIL import Image
import numpy as np

def annotate_image(
    image_path: str,
    detections: sv.Detections,
    labels: list[str]
) -> Image.Image:
    """
    Draws segmentation masks, bounding boxes, and labels on an image.

    Args:
        image_path (str): The path to the source image.
        detections (sv.Detections): The supervision Detections object,
                                    expected to contain `mask` attribute.
        labels (list[str]): The list of string labels for each detection.

    Returns:
        Image.Image: The annotated image as a PIL Image object.
    """
    # Read image from path using OpenCV
    image_bgr = cv2.imread(image_path) # OpenCV reads in BGR format
    
    # Check if image was loaded successfully
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image from path: {image_path}")

    # Create annotators
    # We use ColorLookup.INDEX because Grounding DINO doesn't give class_ids
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    # --- NEW: Layer the annotations ---
    # 1. Apply masks first (they are opaque, so should be at the bottom layer)
    annotated_image_np = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
    
    # 2. Then apply bounding boxes
    annotated_image_np = box_annotator.annotate(scene=annotated_image_np, detections=detections)
    
    # 3. Finally, apply labels on top
    annotated_image_np = label_annotator.annotate(scene=annotated_image_np, detections=detections, labels=labels)

    # Convert the annotated image from OpenCV's BGR format back to a PIL Image (RGB)
    return Image.fromarray(cv2.cvtColor(annotated_image_np, cv2.COLOR_BGR2RGB))