import argparse
import random
import supervision as sv

# Import all your custom classes and the annotate function from utils
from dataset import CamVidDataset
from detector import Detector as GroundingDinoDetector
from yolo_detector import YOLODetector
from segmenter import Segmenter
from utils import annotate_image

def main(args):
    # --- INITIALIZATION ---
    print("Initializing models and dataset...")
    dino_detector = GroundingDinoDetector()
    yolo_detector = YOLODetector()
    sam_segmenter = Segmenter()
    dataset = CamVidDataset(image_dir=args.image_dir, mask_dir=args.mask_dir)
    print("Models and dataset initialized.")

    # --- SELECT RANDOM IMAGE ---
    random_index = random.randint(0, len(dataset) - 1)
    sample = dataset[random_index]
    image_path = sample['image_path']
    print(f"\nProcessing random image #{random_index}: {image_path}")

    # --- PIPELINE 1: GROUNDING DINO -> SAM2 ---
    print("\n--- Running Pipeline A (DINO -> SAM2) ---")
    # dino_prompt = "person, bicycle, car, motorcycle, bus, truck"
    try:
        dino_detections, dino_labels = dino_detector.predict(
            image_path, box_threshold=0.25, text_threshold=0.25
        )

        if len(dino_detections) > 0:
            masks = sam_segmenter.segment(image_path, boxes=dino_detections.xyxy)
            dino_detections.mask = masks
            
            annotated_image_dino = annotate_image(
                image_path=image_path,
                detections=dino_detections,
                labels=dino_labels
            )
            output_path_dino = "output_dino_pipeline.jpg"
            annotated_image_dino.save(output_path_dino)
            print(f"DINO pipeline output saved to: {output_path_dino}")
        else:
            print("DINO found no objects in this image.")

    except Exception as e:
        print(f"An error occurred in the DINO pipeline: {e}")


    # --- PIPELINE 2: YOLOv11 -> SAM2 ---
    print("\n--- Running Pipeline B (YOLOv11 -> SAM2) ---")
    try:
        yolo_detections = yolo_detector.predict(image_path, conf_threshold=0.4)
        
        if len(yolo_detections) > 0:
            masks = sam_segmenter.segment(image_path, boxes=yolo_detections.xyxy)
            yolo_detections.mask = masks
            
            # Create string labels from the class IDs returned by YOLO
            yolo_labels = [
                f"{yolo_detector.class_names[class_id]} {confidence:.2f}"
                for class_id, confidence
                in zip(yolo_detections.class_id, yolo_detections.confidence)
            ]
            
            annotated_image_yolo = annotate_image(
                image_path=image_path,
                detections=yolo_detections,
                labels=yolo_labels
            )
            output_path_yolo = "output_yolo_pipeline.jpg"
            annotated_image_yolo.save(output_path_yolo)
            print(f"YOLO pipeline output saved to: {output_path_yolo}")
        else:
            print("YOLO found no objects in this image.")

    except Exception as e:
        print(f"An error occurred in the YOLO pipeline: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize pipeline outputs on a random CamVid image.")
    parser.add_argument("--image-dir", required=True, help="Path to the CamVid images directory.")
    parser.add_argument("--mask-dir", required=True, help="Path to the CamVid masks directory.")
    args = parser.parse_args()
    main(args)