# In evaluate.py

import argparse
import numpy as np
from tqdm import tqdm

from dataset import CamVidDataset
from detector import Detector as GroundingDinoDetector
from yolo_detector import YOLODetector
from segmenter import Segmenter
from utils import calculate_dice_score

def main(args):
    # --- INITIALIZATION ---
    print("Initializing models and dataset...")
    dino_detector = GroundingDinoDetector()
    yolo_detector = YOLODetector()
    sam_segmenter = Segmenter()
    dataset = CamVidDataset(image_dir=args.image_dir, mask_dir=args.mask_dir)
    
    # Use the --limit argument to run on a subset of the data for quick tests
    limit = args.limit if args.limit is not None else len(dataset)
    
    dino_dice_scores = []
    yolo_dice_scores = []

    print(f"Running evaluation on {limit} images...")

    # --- MAIN EVALUATION LOOP ---
    for i in tqdm(range(limit)):
        sample = dataset[i]
        image_path = sample['image_path']
        gt_mask = sample['ground_truth_mask']

        # Skip images that have no target objects in the ground truth
        if np.sum(gt_mask) == 0:
            continue

        # --- Pipeline 1: Grounding DINO -> SAM ---
        try:
            dino_detections, _ = dino_detector.predict(image_path)
            if len(dino_detections) > 0:
                masks = sam_segmenter.segment(image_path, boxes=dino_detections.xyxy)
                dino_detections.mask = masks
                
                # Combine all predicted instance masks into a single semantic mask
                predicted_mask_dino = np.any(dino_detections.mask, axis=0)
                score = calculate_dice_score(predicted_mask_dino, gt_mask)
                dino_dice_scores.append(score)
        except Exception as e:
            print(f"Error processing {image_path} with DINO pipeline: {e}")

        # --- Pipeline 2: YOLOv8 -> SAM ---
        try:
            yolo_detections = yolo_detector.predict(image_path, conf_threshold=0.5)
            if len(yolo_detections) > 0:
                masks = sam_segmenter.segment(image_path, boxes=yolo_detections.xyxy)
                yolo_detections.mask = masks

                # Combine all predicted instance masks into a single semantic mask
                predicted_mask_yolo = np.any(yolo_detections.mask, axis=0)
                score = calculate_dice_score(predicted_mask_yolo, gt_mask)
                yolo_dice_scores.append(score)
        except Exception as e:
            print(f"Error processing {image_path} with YOLO pipeline: {e}")

    # --- REPORTING RESULTS ---
    avg_dino_score = np.mean(dino_dice_scores) if dino_dice_scores else 0
    avg_yolo_score = np.mean(yolo_dice_scores) if yolo_dice_scores else 0

    print("\n--- Evaluation Complete ---")
    print(f"Processed {len(dino_dice_scores)} images for DINO and {len(yolo_dice_scores)} for YOLO.")
    print(f"Average Dice Score (Grounding DINO -> SAM): {avg_dino_score:.4f}")
    print(f"Average Dice Score (YOLOv8 -> SAM): {avg_yolo_score:.4f}")
    print("-------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation pipelines on the CamVid dataset.")
    parser.add_argument("--image-dir", required=True, help="Path to the CamVid images directory.")
    parser.add_argument("--mask-dir", required=True, help="Path to the CamVid masks directory.")
    parser.add_argument("--limit", type=int, help="Optional: Limit the number of images to process for a quick test.")
    args = parser.parse_args()
    main(args)