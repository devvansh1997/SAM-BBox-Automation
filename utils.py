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

def calculate_dice_score(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """
    Calculates the Dice Score between two binary masks.

    Args:
        predicted_mask (np.ndarray): The binary mask predicted by the model.
        ground_truth_mask (np.ndarray): The binary ground truth mask.

    Returns:
        float: The Dice Score, ranging from 0 (no overlap) to 1 (perfect overlap).
    """
    # Ensure masks are boolean
    predicted_mask = predicted_mask.astype(bool)
    ground_truth_mask = ground_truth_mask.astype(bool)

    # Calculate intersection and the total number of pixels in both masks
    intersection = np.sum(predicted_mask & ground_truth_mask)
    total_pixels = np.sum(predicted_mask) + np.sum(ground_truth_mask)

    # Handle the edge case where both masks are empty
    if total_pixels == 0:
        return 1.0  # Perfect score if there was nothing to segment and we predicted nothing

    # Calculate the Dice Score
    dice_score = (2.0 * intersection) / total_pixels
    return dice_score

def ensemble_detections(
    dino_detections: sv.Detections,
    owl_detections: sv.Detections,
    iou_threshold: float = 0.5
) -> (sv.Detections, list[str]):
    """
    Ensembles detections from two models based on IoU agreement.

    Args:
        dino_detections (sv.Detections): Detections from the first model.
        owl_detections (sv.Detections): Detections from the second model.
        iou_threshold (float): The IoU threshold to consider boxes as a match.

    Returns:
        sv.Detections: A new Detections object with the merged, high-confidence boxes.
        list[str]: The corresponding labels for the ensembled detections.
    """
    if len(dino_detections) == 0 or len(owl_detections) == 0:
        return sv.Detections.empty(), []

    # Calculate the IoU between all pairs of boxes
    iou_matrix = sv.box_iou_batch(dino_detections.xyxy, owl_detections.xyxy)

    # Find matches where IoU is above the threshold
    matches = np.where(iou_matrix >= iou_threshold)
    dino_match_indices, owl_match_indices = matches[0], matches[1]

    # Create a set of owl indices that have been matched to avoid duplicates
    matched_owl_indices = set(owl_match_indices)

    ensembled_boxes = []
    ensembled_scores = []
    ensembled_labels = []

    # For each match, average the boxes and scores
    for dino_idx, owl_idx in zip(dino_match_indices, owl_match_indices):
        # Average the box coordinates
        avg_box = (dino_detections.xyxy[dino_idx] + owl_detections.xyxy[owl_idx]) / 2
        ensembled_boxes.append(avg_box)

        # Average the confidence scores
        avg_score = (dino_detections.confidence[dino_idx] + owl_detections.confidence[owl_idx]) / 2
        ensembled_scores.append(avg_score)
        
        # We can just take the label from one of the models (e.g., DINO)
        # Note: You need to pass the labels into this function if they are not part of the Detections object.
        # Let's assume we pass them separately.
        
    # This is a placeholder for labels. We'll adjust the main script to pass them.
    # For now, let's just return the detections.

    if not ensembled_boxes:
        return sv.Detections.empty(), []

    ensembled_detections = sv.Detections(
        xyxy=np.array(ensembled_boxes),
        confidence=np.array(ensembled_scores)
    )
    
    # We will handle labels in the main script for simplicity
    return ensembled_detections, []