import torch
import numpy as np
import open_clip
from PIL import Image
import logging

log = logging.getLogger(__name__)

class OpenCLIP:
    def __init__(self, device):
        self.device = device

        # Load OpenCLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')


    def extract_jersey_number(self, image):
        image = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
        text_inputs = self.tokenizer([f"jersey number {i}" for i in range(1, 100)]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            confidence, index = similarity[0].max(dim=0)
            jersey_number = index.item() + 1  # since index is 0-based

        return str(jersey_number), confidence.item()

    def process(self, image):
        jersey_number, confidence = self.extract_jersey_number(image)
        return jersey_number, confidence