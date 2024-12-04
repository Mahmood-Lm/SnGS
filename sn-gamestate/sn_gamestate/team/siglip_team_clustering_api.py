# import pandas as pd
# import torch
# import numpy as np
# import logging
# import warnings
# from tracklab.pipeline.videolevel_module import VideoLevelModule
# from .SLip import TeamClassifier
# warnings.filterwarnings("ignore")


# log = logging.getLogger(__name__)


# class TrackletTeamClustering(VideoLevelModule):
#     """
#     This module performs team classification on the embeddings of the tracklets to cluster the detections with role "player" into two teams.
#     Teams are labeled as 0 and 1, and transformer into 'left' and 'right' in a separate module.
#     """
#     input_columns = ["track_id", "embeddings", "role"]
#     output_columns = ["team_cluster"]
    
#     def __init__(self, **kwargs):
#         super().__init__()
        
#     @torch.no_grad()
#     def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):

#         # Print the columns of the detections DataFrame
#         # print("Available columns in detections DataFrame:", detections.columns)
#         print("available columns in metadatas DataFrame:", metadatas.file_path)

#         player_detections = detections[detections.role == "player"]
#         print("player_detections:", player_detections.embeddings.values)

#         if player_detections.empty:
#             detections['team_cluster'] = np.nan  # Initialize 'team_cluster' with a default value
#             return detections

#         # Sample 10% of all player detections for fitting the model
#         sample_size = int(0.1 * len(player_detections))
#         sample_detections = player_detections.sample(n=sample_size, random_state=0)

#         # Extract embeddings from the sampled detections
#         sample_embeddings = np.vstack(sample_detections.embeddings.values)

#         # Initialize the TeamClassifier
#         team_classifier = TeamClassifier(batch_size=250, device='cuda')

#         # Save a grid image of 100 player crops
#         team_classifier.save_image_grid(sample_embeddings, 'output/grid_image.png')

#         # Fit the model on the sampled embeddings
#         team_classifier.fit(sample_embeddings)

#         # Predict the team clusters for all player detections
#         all_embeddings = np.vstack(player_detections.embeddings.values)
#         player_detections['team_cluster'] = team_classifier.predict(all_embeddings)

#         # Map the team cluster back to the original detections DataFrame
#         detections = detections.merge(player_detections[['track_id', 'team_cluster']], on='track_id', how='left', sort=False)

#         return detections


import pandas as pd
import torch
import numpy as np
import logging
import warnings
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule
from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel
import umap
from tqdm import tqdm
import cv2
from PIL import Image
from typing import List, TypeVar

warnings.filterwarnings("ignore")

V = TypeVar("V")

log = logging.getLogger(__name__)

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'

def create_batches(sequence, batch_size):
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch

class TeamClassifier:
    def __init__(self, device: str = 'cuda', batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        # Convert OpenCV images to PIL images
        crops = [Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
        return np.concatenate(data)

    def fit(self, crops):
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops):
        if len(crops) == 0:
            return np.array([])
        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)

class TrackletTeamClustering(DetectionLevelModule):
    input_columns = ["track_id", "embeddings", "role"]
    output_columns = ["team_cluster"]

    def __init__(self, cfg, device, batch_size, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        self.team_classifier = TeamClassifier(device=device, batch_size=batch_size)
        self.device = device
        self.cfg = cfg


    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]

        # Double the size of the cropped image for better recognition
        crop = Image.fromarray(crop).resize((crop.shape[1] * 2, crop.shape[0] * 2))
        crop = np.array(crop)

        return crop

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        player_detections = detections[detections.role == "player"]

        if player_detections.empty:
            detections['team_cluster'] = np.nan  # Initialize 'team_cluster' with a default value
            return detections

        # Crop the players from the image
        crops = [self.preprocess(batch['img'][i], row, metadatas.iloc[i]) for i, row in player_detections.iterrows()]

        # Select a random subset (10%) of the crops for fitting the model
        subset_size = max(1, int(0.1 * len(crops)))
        subset_indices = np.random.choice(len(crops), subset_size, replace=False)
        subset_crops = [crops[i] for i in subset_indices]

        # Perform clustering using TeamClassifier
        self.team_classifier.fit(subset_crops)
        player_detections['team_cluster'] = self.team_classifier.predict(crops)

        # Map the team cluster back to the original detections DataFrame
        detections = detections.merge(player_detections[['track_id', 'team_cluster']], on='track_id', how='left', sort=False)

        return detections