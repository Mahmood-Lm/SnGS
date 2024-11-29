# import torch
# import numpy as np
# import open_clip
# from PIL import Image
# import logging
# from torchvision import transforms


# log = logging.getLogger(__name__)

# class OpenCLIP:
#     def __init__(self, device):
#         self.device = device
#         self.pretrained_path = '/home/Mahmood/soccernet/sn-gamestate/pretrained_models/jersey/epoch_5.pt'

#         # Load OpenCLIP model
#         self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained="openai", device=self.device)
#         self.tokenizer = open_clip.get_tokenizer('ViT-L-14')


#     def is_jersey_number_visible(self, image):
#         image = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
#         text_inputs = self.tokenizer(["visible jersey number", "hidden jersey number"]).to(self.device)

#         with torch.no_grad():
#             image_features = self.model.encode_image(image)
#             text_features = self.model.encode_text(text_inputs)

#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             text_features /= text_features.norm(dim=-1, keepdim=True)

#             similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#             confidence, index = similarity[0].max(dim=0)
#             is_visible = index.item() == 0  # 0 for visible, 1 for not visible

#         return is_visible

#     def extract_jersey_number(self, image):
#         image = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
#         text_inputs = self.tokenizer([f"jersey number {i}" for i in range(2, 100)]).to(self.device)

#         with torch.no_grad():
#             image_features = self.model.encode_image(image)
#             text_features = self.model.encode_text(text_inputs)

#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             text_features /= text_features.norm(dim=-1, keepdim=True)

#             similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
#             confidence, index = similarity[0].max(dim=0)
#             jersey_number = index.item() + 2  # Adjust for "none"

#         if jersey_number is None:
#             return None, 0
#         else:
#             return str(jersey_number), confidence.item()

#     def process(self, image):
#         # if self.is_jersey_number_visible(image):
#         if 1>0:
#             jersey_number, confidence = self.extract_jersey_number(image)
#         else:
#             jersey_number, confidence = None, 0
#         return jersey_number, confidence






#################################################################################################################

import torch
import numpy as np
import open_clip
from PIL import Image
import logging

log = logging.getLogger(__name__)

class OpenCLIP:
    def __init__(self, device):
        self.device = device
        self.pretrained_path = '/home/Mahmood/soccernet/sn-gamestate/pretrained_models/jersey/V3_epoch_5.pt'
        # self.pretrained_path = '/home/Mahmood/soccernet/sn-gamestate/wise-ft/wft_models/wft_model_0.5.pt'


        # Load OpenCLIP model
        # self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained="openai", device=self.device)

        #Finetuned model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None, device=self.device)

        checkpoint = torch.load(self.pretrained_path, map_location=self.device)

        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)



        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')


    def extract_jersey_number(self, image):
        image = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
        # text_inputs = self.tokenizer([f"jersey number {i}" for i in range(2, 100)]).to(self.device)
        text_inputs = self.tokenizer([f"jersey number {i}" for i in range(2, 100)] + ["jersey number none"]).to(self.device)


        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            confidence, index = similarity[0].max(dim=0)
            # jersey_number = index.item() + 2  # since index is 0-based
            jersey_number = (index.item() + 2) if index.item() < 98 else None  # Adjust for "none"

        if jersey_number is None:
            return None, 0
        else:
            return str(jersey_number), confidence.item()

    def process(self, image):
        jersey_number, confidence = self.extract_jersey_number(image)
        return jersey_number, confidence