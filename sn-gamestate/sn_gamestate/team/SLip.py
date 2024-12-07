from typing import Generator, Iterable, List, TypeVar

import numpy as np
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
from PIL import Image
import cv2
import os

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


def cv2_to_pillow(image: np.ndarray) -> Image.Image:
    """
    Convert an OpenCV image to a Pillow image.

    Args:
        image (np.ndarray): The OpenCV image.

    Returns:
        Image.Image: The Pillow image.
    """
    # if image.dtype != np.uint8:
    #     if image.max() <= 1.0:
    #         image = (image * 255).astype(np.uint8)
    #     else:
    #         image = image.astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
       """
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)


    def save_image_grid(self, crops: List[np.ndarray], output_path: str) -> None:
        """
        Save a grid of 100 images from the list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
            output_path (str): Path to save the image grid.
        """
        if len(crops) < 100:
            raise ValueError("Not enough images to create a grid of 100 images.")

        # Select the first 100 images
        selected_crops = crops[:100]
        pillow_images = [cv2_to_pillow(crop) for crop in selected_crops]

        # Create a grid of 10x10 images
        grid_size = 10
        image_size = pillow_images[0].size
        grid_image = Image.new('RGB', (grid_size * image_size[0], grid_size * image_size[1]))

        for i, img in enumerate(pillow_images):
            row = i // grid_size
            col = i % grid_size
            grid_image.paste(img, (col * image_size[0], row * image_size[1]))

        # Save the grid image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        grid_image.save(output_path)