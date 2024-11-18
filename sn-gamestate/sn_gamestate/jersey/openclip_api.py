import pandas as pd
import torch
import numpy as np
import open_clip
from PIL import Image
import logging
import gc

from tracklab.utils.collate import default_collate, Unbatchable
from tracklab.pipeline.detectionlevel_module import DetectionLevelModule

log = logging.getLogger(__name__)

class OpenCLIP(DetectionLevelModule):
    input_columns = []
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]
    collate_fn = default_collate

    def __init__(self, cfg, device, batch_size, tracking_dataset=None):
        super().__init__(batch_size=batch_size)
        self.cfg = cfg
        self.device = device
        self.model = None
        self.preprocess1 = None
        self.tokenizer = None
        self.pretrained_path = '/home/Mahmood/soccernet/sn-gamestate/pretrained_models/jersey/epoch_5.pt'


    # Load the model and tokenizer when the jersey recognition module is needed as to not overload the memory.
    def load_model(self):
        if self.model is None:
            self.model, _, self.preprocess1 = open_clip.create_model_and_transforms('ViT-L-14', pretrained="openai" , device=self.device)


            # print("Device:", self.device)
            # Print initial parameters
            print("Initial parameters:", list(self.model.parameters())[0][0][:5])

            # try:
            #     # Load the fine-tuned model
            #     checkpoint = torch.load(self.pretrained_path, map_location=self.device)

            #     if 'state_dict' in checkpoint:
            #         self.model.load_state_dict(checkpoint['state_dict'])
            #     else:
            #         self.model.load_state_dict(checkpoint)

            #     # self.model.to(self.device)

            #     # Print parameters after loading checkpoint
            #     print("Parameters after loading checkpoint:", list(self.model.parameters())[0][0][:5])
            # except Exception as e:
            #     print(f"Error loading fine-tuned model: {e}")
            #     return
            
            # self.model.to(self.device)
            self.tokenizer = open_clip.get_tokenizer('ViT-L-14')

    def no_jersey_number(self):
        return [None, None, 0]

    @torch.no_grad()
    def preprocess(self, image, detection: pd.Series, metadata: pd.Series):
        l, t, r, b = detection.bbox.ltrb(image_shape=(image.shape[1], image.shape[0]), rounded=True)
        crop = image[t:b, l:r]

        # Double the size of the cropped image for better recognition
        crop = Image.fromarray(crop).resize((crop.shape[1] * 2, crop.shape[0] * 2))
        crop = np.array(crop)

        crop = Unbatchable([crop])
        batch = {
            "img": crop,
        }
<<<<<<< HEAD
        return self.preprocess(batch)

    def extract_jersey_numbers_from_clip(self, images):
        images = torch.stack([self.preprocess(Image.fromarray(img)).unsqueeze(0) for img in images]).to(self.device)
=======
        return batch

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        self.load_model()  # Ensure the model is loaded before processing the batch
        images_np = [img.cpu().numpy() for img in batch['img']]
        del batch['img']

        jersey_numbers, confidences = self.extract_jersey_number(images_np)

        detections["jersey_number_detection"] = jersey_numbers
        detections["jersey_number_confidence"] = confidences
        return detections

    def extract_jersey_number(self, images):
        self.load_model()  # Ensure the model is loaded before processing the batch
        images = torch.stack([self.preprocess1(Image.fromarray(image)) for image in images]).to(self.device)
<<<<<<< HEAD
        # text_inputs = self.tokenizer([f"jersey number {i}" for i in range(2, 100)]).to(self.device)
        text_inputs = self.tokenizer([f"jersey number {i}" for i in range(2, 100)] + ["jersey number none"]).to(self.device)

=======
>>>>>>> 28dfcdfe75f8688ebe98e690e78e92c12f730539
        text_inputs = self.tokenizer([f"jersey number {i}" for i in range(1, 100)]).to(self.device)
>>>>>>> e3c5b2928d5818e71290cf3f89a8b271abf46a0e

        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            confidences, indices = similarity.max(dim=-1)
<<<<<<< HEAD
            # jersey_numbers = (indices + 2).tolist()  # since index is 0-based and we start from 2
            jersey_numbers = [(index + 2) if index < 98 else None for index in indices.tolist()]  # Adjust for "none"

=======
<<<<<<< HEAD
            jersey_numbers = indices.cpu().numpy() + 1  # since index is 0-based

        return jersey_numbers, confidences.cpu().numpy()

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        jersey_number_detection = []
        jersey_number_confidence = []
        images_np = [img.cpu().numpy() for img in batch['img']]
        del batch['img']

        jersey_numbers, confidences = self.extract_jersey_numbers_from_clip(images_np)
        for jn, conf in zip(jersey_numbers, confidences):
            jersey_number_detection.append(str(jn))
            jersey_number_confidence.append(conf)

        detections['jersey_number_detection'] = jersey_number_detection
        detections['jersey_number_confidence'] = jersey_number_confidence

        return detections
=======
            jersey_numbers = (indices + 1).tolist()  # since index is 0-based
>>>>>>> e3c5b2928d5818e71290cf3f89a8b271abf46a0e

            # Filter results based on confidence threshold
            filtered_jersey_numbers = []
            filtered_confidences = []
            for jersey_number, confidence in zip(jersey_numbers, confidences.tolist()):
                if jersey_number is not None and confidence >= 0.1:
                    filtered_jersey_numbers.append(str(jersey_number))
                    filtered_confidences.append(confidence)
                    print(f"Jersey number: {jersey_number}, Confidence: {confidence}")
                else:
                    filtered_jersey_numbers.append(None)
                    filtered_confidences.append(0)

        # Clear memory
        del images, text_inputs, image_features, text_features, similarity
        gc.collect()
        torch.cuda.empty_cache()
            

        return filtered_jersey_numbers, filtered_confidences
>>>>>>> 28dfcdfe75f8688ebe98e690e78e92c12f730539
