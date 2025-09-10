from detector import Detector
from segmenter import Segmenter
from utils import annotate_image
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects in an image")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--prompt", default="person . car . bus . motorcycle", help="Text prompt to detect classes")
    parser.add_argument("--output", default="output.jpg", help="Path for output file")
    args = parser.parse_args()

    # initialize the detector
    dino = Detector()

    # intialize the segmentor
    sam2 = Segmenter()

    # get predictions for bounding boxes
    detections, labels = dino.predict(
        image_path=args.image,
        text_prompt=args.prompt
    )

    # if objects are found
    if len(detections) > 0:

        # get segmentation results back
        masks = sam2.segment(
            image_path=args.image,
            boxes=detections.xyxy
        )

        # attach new found masks to detections object
        detections.mask = masks
    

    # annotate image with predictions
    annotate_image = annotate_image(
        image_path=args.image,
        detections=detections,
        labels=labels
    )

    # save the results
    annotate_image.save(args.output)
    print(f"---- Annotated Image saved to {args.output} ----")