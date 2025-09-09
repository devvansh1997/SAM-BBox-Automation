import torch
import supervision as sv
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import config

class Detector:
    def __init__(self):
        """
        loading the model should be done once to increase speed, 
        so here I create a class for the detector
        """
        print("---- Loading Detection Model ----")
        # load model on GPU
        self.device = torch.device(config.DEVICE)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(config.DET_MODEL_ID).to(self.device)
        self.processor = AutoProcessor.from_pretrained(config.DET_MODEL_ID)
        print(f"---- Model {config.DET_MODEL_ID} has been loaded on {config.DEVICE} ----")
    
    def predict(
            self, 
            image_path: str, 
            text_prompt: str = "person . car . bus . motorcycle",
            box_threshold: float = config.BOX_THRESHOLD,
            text_threshold: float = config.TEXT_THRESHOLD
            ) -> tuple[sv.Detections, list[str]]:
        """
        Performs inference on a single local image.

        Args:
            image_path (str): The path to the local image file.
            text_prompt (str): The text prompt for detection.
            box_threshold (float, optional): The box confidence threshold. Defaults to config.BOX_THRESHOLD.
            text_threshold (float, optional): The text association threshold. Defaults to config.TEXT_THRESHOLD.

        Returns:
            tuple[sv.Detections, list[str]]: A tuple containing the detections and their corresponding labels.
        """

        # load the image using PIL
        image = Image.open(image_path).convert("RGB")

        # prepare images using AutoModelForZeroShotObjectDetection from hf
        inputs = self.processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        # peform the inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # post process results
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        # convert results to standardized format
        detections = sv.Detections(
            xyxy=results["boxes"].cpu().numpy(),
            confidence=results["scores"].cpu().numpy(),
        )
        labels = results["text_labels"]

        return detections, labels
