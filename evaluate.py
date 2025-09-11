import argparse
import numpy as np
from tqdm import tqdm
import supervision as sv

from dataset import CamVidDataset
from detector import Detector as GroundingDinoDetector
from yolo_detector import YOLODetector
from segmenter import Segmenter
from utils import calculate_dice_score, ensemble_detections

def main(args):
    # --- INITIALIZATION ---
    dino_detector = GroundingDinoDetector()
    yolo_detector = YOLODetector()
    sam_segmenter = Segmenter()
    dataset = CamVidDataset(image_dir=args.image_dir, mask_dir=args.mask_dir)
    
    limit = args.limit if args.limit is not None else len(dataset)
    
    # Create lists to store scores for each pipeline
    dino_dice_scores = []
    yolo_dice_scores = []
    
    prompt = "person, bicycle, car, motorcycle, bus, truck"
    print(f"Running evaluation on {limit} images...")

    # --- MAIN EVALUATION LOOP ---
    for i in tqdm(range(limit)):
        sample = dataset[i]
        image_path = sample['image_path']
        gt_mask = sample['ground_truth_mask']

        if np.sum(gt_mask) == 0:
            continue

        # --- Run all detectors once per image for efficiency ---
        try:
            dino_detections, _ = dino_detector.predict(image_path, prompt, box_threshold=0.25, text_threshold=0.25)
        except Exception as e:
            print(f"Error in DINO detection: {e}")
            dino_detections = sv.Detections.empty()
            
        try:
            yolo_detections = yolo_detector.predict(image_path, conf_threshold=0.4)
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            yolo_detections = sv.Detections.empty()

        # --- Pipeline A (Baseline): Grounding DINO -> SAM2 ---
        if len(dino_detections) > 0:
            masks = sam_segmenter.segment(image_path, boxes=dino_detections.xyxy)
            if masks.any():
                dino_detections.mask = masks
                predicted_mask = np.any(dino_detections.mask, axis=0)
                score = calculate_dice_score(predicted_mask, gt_mask)
                dino_dice_scores.append(score)

        # --- Pipeline B (Invention 2): YOLOv8 -> SAM2 ---
        if len(yolo_detections) > 0:
            masks = sam_segmenter.segment(image_path, boxes=yolo_detections.xyxy)
            if masks.any():
                yolo_detections.mask = masks
                predicted_mask = np.any(yolo_detections.mask, axis=0)
                score = calculate_dice_score(predicted_mask, gt_mask)
                yolo_dice_scores.append(score)

    # --- REPORTING RESULTS ---
    avg_dino_score = np.mean(dino_dice_scores) if dino_dice_scores else 0
    avg_yolo_score = np.mean(yolo_dice_scores) if yolo_dice_scores else 0

    print("\n--- Evaluation Complete ---")
    print(f"Processed {limit} of {len(dataset)} images.")
    print("\n--- Average Dice Scores ---")
    print(f"Pipeline A (Baseline: DINO -> SAM2):    {avg_dino_score:.4f}")
    print(f"Pipeline B (New: YOLOv11 -> SAM2):   {avg_yolo_score:.4f}")
    print("-------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate segmentation pipelines on the CamVid dataset.")
    parser.add_argument("--image-dir", required=True, help="Path to the CamVid images directory.")
    parser.add_argument("--mask-dir", required=True, help="Path to the CamVid masks directory.")
    parser.add_argument("--limit", type=int, required=False, help="Optional: Limit the number of images to process for a quick test.")
    args = parser.parse_args()
    main(args)