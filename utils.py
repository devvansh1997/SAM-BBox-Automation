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
    Draws bounding boxes and labels on an image.

    Args:
        image_path (str): The path to the source image.
        detections (sv.Detections): The supervision Detections object.
        labels (list[str]): The list of string labels for each detection.

    Returns:
        Image.Image: The annotated image as a PIL Image object.
    """
    # Read image from path using OpenCV, as supervision annotators work best with NumPy arrays
    image = cv2.imread(image_path)

    # Create annotators
    box_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

    # Annotate the image
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # Convert the annotated image from OpenCV's BGR format back to a PIL Image (RGB)
    return Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    