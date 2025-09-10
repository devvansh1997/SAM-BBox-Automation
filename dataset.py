# import os
from PIL import Image
import numpy as np
import os

# CamVid Class Colors (RGB) - a mapping of class names to their colors in the masks
# You can find these in the dataset's documentation or supporting files.
CAMVID_CLASS_COLORS = {
    'Bicyclist': (192, 128, 128),
    'Car': (64, 0, 128),
    'Pedestrian': (64, 64, 0),
    'Truck': (0, 0, 64),
    'Motorbike': (192, 0, 128),
    # Add other classes if needed, but we'll filter for the ones below
}

class CamVidDataset:
    def __init__(self, image_dir: str, mask_dir: str):
        """
        Initializes the CamVid Dataset handler.

        Args:
            image_dir (str): Path to the directory containing the images.
            mask_dir (str): Path to the directory containing the ground truth masks.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # Get a sorted list of all image filenames
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        # Define our target classes for the experiment
        self.target_classes = {
            'person': ['Pedestrian', 'Bicyclist'],
            'vehicle': ['Car', 'Truck', 'Motorbike']
        }
        
        # Get the corresponding RGB colors for our target classes
        self.target_colors = [CAMVID_CLASS_COLORS[cls] for group in self.target_classes.values() for cls in group]

    def _process_mask(self, mask_path: str) -> np.ndarray:
        """
        Converts a color-coded CamVid mask into a binary mask for target classes.

        Args:
            mask_path (str): The file path to the ground truth mask.

        Returns:
            np.ndarray: A binary mask (0s and 1s) where 1 indicates a target class pixel.
        """
        # Load the ground truth mask
        mask_img = Image.open(mask_path).convert("RGB")
        mask_np = np.array(mask_img)
        
        # Create an empty, all-black (zeros) mask to start with
        binary_mask = np.zeros((mask_np.shape[0], mask_np.shape[1]), dtype=np.uint8)
        
        # For each of our target colors, find where they are in the mask and set those
        # pixels to 1 (white) in our binary mask.
        for color in self.target_colors:
            target = np.all(mask_np == color, axis=-1)
            binary_mask[target] = 1
            
        return binary_mask

    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset at a given index.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the image path and its binary ground truth mask.
        """
        # Get the image filename
        img_filename = self.image_filenames[index]
        image_path = os.path.join(self.image_dir, img_filename)
        
        # Construct the corresponding mask filename (e.g., '0001TP_006690.png' -> '0001TP_006690_L.png')
        mask_filename = img_filename.replace('.png', '_L.png')
        mask_path = os.path.join(self.mask_dir, mask_filename)
        
        # Process the mask to get our binary ground truth
        ground_truth_mask = self._process_mask(mask_path)
        
        return {
            'image_path': image_path,
            'ground_truth_mask': ground_truth_mask
        }