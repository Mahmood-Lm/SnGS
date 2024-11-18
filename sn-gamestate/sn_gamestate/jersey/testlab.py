import pandas as pd
import torch
import numpy as np
import cv2
from mmocr_api import MMOCR
from mmocr.apis import TextDetInferencer, TextRecInferencer
from mmocr.apis import MMOCRInferencer
from testclip import OpenCLIP

device = 'cuda'

#MMOCR
# ocr = MMOCRInferencer(det='dbnet_resnet18_fpnc_1200e_icdar2015', rec='SAR')
# textdetinferencer = TextDetInferencer('dbnet_resnet18_fpnc_1200e_icdar2015', device=device)
# textrecinferencer = TextRecInferencer('SAR', device=device)

#CLIP
model = OpenCLIP(device=device)



#ImagePrep
image = cv2.imread('/home/Mahmood/soccernet/sn-gamestate/sn_gamestate/jersey/images/164_229.jpg')
image_np = np.array(image)
image_np = cv2.resize(image_np, (image_np.shape[1] * 2, image_np.shape[0] * 2))

#Run CLIP
jersey_number, confidence = model.process(image_np)
print("CLIP Results: ",jersey_number, confidence)

#Run MMOCR
# result = ocr(image_np)
# print("MMOCR Results",result)


