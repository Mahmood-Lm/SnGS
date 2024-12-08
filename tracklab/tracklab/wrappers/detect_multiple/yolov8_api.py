# import os
# import torch
# import pandas as pd
# import numpy as np

# from typing import Any
# from tracklab.pipeline.imagelevel_module import ImageLevelModule

# os.environ["YOLO_VERBOSE"] = "False"
# from ultralytics import YOLO

# from tracklab.utils.coordinates import ltrb_to_ltwh

# import logging

# log = logging.getLogger(__name__)


# def collate_fn(batch):
#     idxs = [b[0] for b in batch]
#     images = [b["image"] for _, b in batch]
#     shapes = [b["shape"] for _, b in batch]
#     return idxs, (images, shapes)


# class YOLOv8(ImageLevelModule):
#     collate_fn = collate_fn
#     input_columns = []
#     output_columns = [
#         "image_id",
#         "video_id",
#         "category_id",
#         "bbox_ltwh",
#         "bbox_conf",
#     ]

#     def __init__(self, cfg, device, batch_size, **kwargs):
#         super().__init__(batch_size)
#         self.cfg = cfg
#         self.device = device
#         self.model = YOLO(cfg.path_to_checkpoint)
#         self.model.to(device)
#         self.id = 0
#         self.prev_frame = None
#         self.default_conf = cfg.min_confidence
#         self.default_iou = cfg.iou
#         self.conf = self.default_conf
#         self.iou = self.default_iou

#     def detect_severe_movement(self, current_frame):
#         if self.prev_frame is None:
#             self.prev_frame = current_frame
#             return False, 0

#         # Calculate the absolute difference between the current frame and the previous frame
#         frame_diff = np.abs(current_frame.astype(np.float32) - self.prev_frame.astype(np.float32))
#         mean_diff = np.mean(frame_diff)

#         # Update the previous frame
#         self.prev_frame = current_frame

#         # Define a threshold for severe movement
#         movement_threshold = 14.5  # Adjust this threshold based on your requirements

#         return mean_diff > movement_threshold, mean_diff

#     @torch.no_grad()
#     def preprocess(self, image, detections, metadata: pd.Series):
#         return {
#             "image": image,
#             "shape": (image.shape[1], image.shape[0]),
#         }

#     @torch.no_grad()
#     def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
#         images, shapes = batch
#         detections = []

#         for image, shape, (_, metadata) in zip(images, shapes, metadatas.iterrows()):
#             # Detect severe camera movement
#             severe_movement, mean_diff = self.detect_severe_movement(image)

#             # Adjust confidence and IoU thresholds based on camera movement
#             if severe_movement:
#                 self.conf = self.default_conf * 0.6  # Decrease confidence threshold
#                 self.iou = self.default_iou * 0.45  # Decrease IoU threshold
#                 print(f"Severe movement detected in frame {metadata.name} with mean_diff {mean_diff:.2f}. Adjusting conf to {self.conf} and iou to {self.iou}.")
#             else:
#                 # Reset to default values if no severe movement is detected
#                 if self.conf != self.default_conf or self.iou != self.default_iou:
#                     print(f"No severe movement detected in frame {metadata.name}. Resetting conf to {self.default_conf} and iou to {self.default_iou}.")
#                 self.conf = self.default_conf
#                 self.iou = self.default_iou

#             results = self.model(image, iou=self.iou)
#             for result in results:
#                 for bbox in result.boxes.cpu().numpy():
#                     if bbox.cls == 0 and bbox.conf >= self.conf:
#                         detections.append(
#                             pd.Series(
#                                 dict(
#                                     image_id=metadata.name,
#                                     bbox_ltwh=ltrb_to_ltwh(bbox.xyxy[0], shape),
#                                     bbox_conf=bbox.conf[0],
#                                     video_id=metadata.video_id,
#                                     category_id=1,
#                                 ),
#                                 name=self.id,
#                             )
#                         )
#                         self.id += 1

#         return detections



import os
import torch
import pandas as pd

from typing import Any
from tracklab.pipeline.imagelevel_module import ImageLevelModule

os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO

from tracklab.utils.coordinates import ltrb_to_ltwh

import logging

log = logging.getLogger(__name__)


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOv8(ImageLevelModule):
    collate_fn = collate_fn
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
    ]

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.model = YOLO(cfg.path_to_checkpoint)
        self.model.to(device)
        self.id = 0
        # self.iou = cfg.iou

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        images, shapes = batch
        results_by_image = self.model(images) #Edited to include imgsz
        detections = []
        for results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            for bbox in results.boxes.cpu().numpy():
                # discard `ball` class
                # if bbox.cls == 0 and bbox.conf >= self.cfg.min_confidence: #Removed discarding of classes as we have only 1 class
                if bbox.conf >= self.cfg.min_confidence: #Removed discarding of classes as we have only 1 class
                    bbox_ltwh = ltrb_to_ltwh(bbox.xyxy[0], shape)
                    bbox_width = bbox_ltwh[2]
                    bbox_height = bbox_ltwh[3]
                    # print(f"Bounding box size - Width: {bbox_width}, Height: {bbox_height}, Confidence: {bbox.conf}")
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltrb_to_ltwh(bbox.xyxy[0], shape),
                                bbox_conf=bbox.conf[0],
                                video_id=metadata.video_id,
                                category_id= 1,  # Classes: 1: 'goalkeeper', 2: 'player', 3: 'referee'
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detections
