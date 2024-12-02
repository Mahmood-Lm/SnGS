import pandas as pd
import torch
import numpy as np
import logging
import warnings
from tracklab.pipeline.videolevel_module import VideoLevelModule
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
import umap
from transformers import AutoProcessor, SiglipVisionModel
import supervision as sv
from tqdm import tqdm

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
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops):
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
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

class TrackletTeamClustering(VideoLevelModule):
    input_columns = ["track_id", "embeddings", "role"]
    output_columns = ["team_cluster"]
    
    def __init__(self, **kwargs):
        super().__init__()
        self.team_classifier = TeamClassifier(device='cuda' if torch.cuda.is_available() else 'cpu')
        
    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        player_detections = detections[detections.role == "player"]

        # Compute mean embeddings for each track_id
        embeddings_list = []
        for track_id, group in player_detections.groupby("track_id"):
            if np.isnan(track_id):
                continue
            embeddings = np.mean(np.vstack(group.embeddings.values), axis=0)
            embeddings_list.append({'track_id': track_id, 'embeddings': embeddings})

        if not embeddings_list:  # Check if embeddings_list is empty
            detections['team_cluster'] = np.nan  # Initialize 'team_cluster' with a default value
            return detections

        embedding_tracklet = pd.DataFrame(embeddings_list)

        if len(embedding_tracklet) == 1:  # Only one track_id and embedding
            embedding_tracklet['team_cluster'] = 0
        else:
            # Perform clustering using TeamClassifier
            embeddings = np.vstack(embedding_tracklet.embeddings.values)
            self.team_classifier.fit(embeddings)
            embedding_tracklet['team_cluster'] = self.team_classifier.predict(embeddings)

        # Map the team cluster back to the original detections DataFrame
        detections = detections.merge(embedding_tracklet[['track_id', 'team_cluster']], on='track_id', how='left', sort=False)

        return detections