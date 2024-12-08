import pandas as pd
import torch
import numpy as np
import logging
import warnings
from tracklab.pipeline.videolevel_module import VideoLevelModule
from .SLip import TeamClassifier
from PIL import Image
import cv2
import os
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# log = logging.getLogger(__name__)

class TrackletTeamClustering(VideoLevelModule):
    """
    This module performs team classification on the crops of the tracklets to cluster the detections with role "player" into two teams.
    Teams are labeled as 0 and 1, and transformer into 'left' and 'right' in a separate module.
    """
    input_columns = ["track_id", "bbox_ltwh", "role", "image_id", "bbox_conf"]
    output_columns = ["team_cluster"]

    def __init__(self, **kwargs):
        super().__init__()

    @torch.no_grad()
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame):
        # Print the columns of the detections DataFrame
        # print("available columns in metadatas DataFrame:", metadatas.file_path)

        # print("Metadatas:" ,metadatas.columns)
        # print("******************")
        # print("Detections:", detections.columns)

        # print(detections.visibility_scores)

        player_detections = detections[detections.role == "player"]
        # print("player_detections:", player_detections)

        if player_detections.empty:
            detections['team_cluster'] = np.nan  # Initialize 'team_cluster' with a default value
            return detections

        # Ensure bbox_conf is numeric
        player_detections['bbox_conf'] = pd.to_numeric(player_detections['bbox_conf'], errors='coerce')

        # Select the top 5% most confident player detections for fitting the model
        sample_size = int(0.05 * len(player_detections))
        sample_detections = player_detections.nlargest(sample_size, 'bbox_conf')

        # Select the top 10% most confident player detections from each image for fitting the model
        # sample_detections = pd.DataFrame()
        # for image_id, group in player_detections.groupby('image_id'):
        #     sample_size = max(1, int(0.1 * len(group)))
        #     top_detections = group.nlargest(sample_size, 'bbox_conf')
        #     sample_detections = pd.concat([sample_detections, top_detections])

        # Extract crops for the sampled player detections
        sample_crops = []
        for _, row in tqdm(sample_detections.iterrows(), total=sample_detections.shape[0], desc="Collecting sample crops", position=0):
            image_path = metadatas.loc[metadatas['id'] == row['image_id'], 'file_path'].values[0]
            image = cv2.imread(image_path)
            l, t, w, h = map(int, row['bbox_ltwh'])
            r, b = l + w, t + h
            crop = image[t:b, l:r]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                crop = np.zeros((10, 10, 3), dtype=np.uint8)
            sample_crops.append(crop)

        # Initialize the TeamClassifier
        team_classifier = TeamClassifier(batch_size=8, device='cuda')

        # Save a grid image of 100 player crops
        # team_classifier.save_image_grid(sample_crops, 'output/fit.png')

        # Fit the model on the sampled crops
        team_classifier.fit(sample_crops)

        # Remove the 10% crops from memory
        del sample_crops

        # Collect crops for the 5% most confident bounding boxes of each tracklet
        tracklet_clusters = []
        all_top_crops = []
        tracklet_indices = []

        for track_id, group in tqdm(player_detections.groupby('track_id'), desc="Collecting tracklet crops"):
            # If the tracklet has less than 6 entries, take all of them
            if len(group) < 6:
                top_detections = group
            else:  
                # Sort by confidence and select the top 5% (at least 5)
                top_detections = group.nlargest(max(5, int(0.05 * len(group))), 'bbox_conf')
            for _, row in top_detections.iterrows():
                image_path = metadatas.loc[metadatas['id'] == row['image_id'], 'file_path'].values[0]
                image = cv2.imread(image_path)
                l, t, w, h = map(int, row['bbox_ltwh'])
                r, b = l + w, t + h
                crop = image[t:b, l:r]
                if crop.shape[0] == 0 or crop.shape[1] == 0:
                    crop = np.zeros((10, 10, 3), dtype=np.uint8)
                all_top_crops.append(crop)
                tracklet_indices.append(track_id)

        # Predict the team clusters for all collected crops
        clusters = team_classifier.predict(all_top_crops)

        # Save a grid image of 100 player crops
        # team_classifier.save_image_grid(all_top_crops, 'output/predict.png')

        # Remove the collected crops from memory
        del all_top_crops

        # Assign the most frequent cluster to each tracklet
        tracklet_cluster_map = {}
        for track_id, cluster in zip(tracklet_indices, clusters):
            if track_id not in tracklet_cluster_map:
                tracklet_cluster_map[track_id] = []
            tracklet_cluster_map[track_id].append(cluster)

        for track_id, cluster_list in tracklet_cluster_map.items():
            most_frequent_cluster = np.bincount(cluster_list).argmax()
            tracklet_clusters.append((track_id, most_frequent_cluster))

        # Create a DataFrame for the tracklet clusters
        tracklet_clusters_df = pd.DataFrame(tracklet_clusters, columns=['track_id', 'team_cluster'])

        # Map the team cluster back to the original detections DataFrame
        detections = detections.merge(tracklet_clusters_df, on='track_id', how='left', sort=False)

        
        del tracklet_clusters, tracklet_cluster_map, tracklet_indices, tracklet_clusters_df

        return detections
