import pandas as pd
import torch
import numpy as np
from mmocr.apis import MMOCRInferencer
from mmocr.apis import TextDetInferencer, TextRecInferencer
from mmocr.utils import bbox2poly, crop_img, poly2bbox
import logging

from tracklab.utils.collate import default_collate, Unbatchable

log = logging.getLogger(__name__)

class simpleMMOCR():
    input_columns = ["image"]
    output_columns = ["jersey_number_detection", "jersey_number_confidence"]

    def __init__(self, batch_size, device, tracking_dataset=None):
        self.ocr = MMOCRInferencer(det='dbnet_resnet18_fpnc_1200e_icdar2015', rec='SAR')
        self.batch_size = batch_size
        self.device = device

        self.textdetinferencer = TextDetInferencer(
            'dbnet_resnet18_fpnc_1200e_icdar2015', device=device)
        self.textrecinferencer = TextRecInferencer('SAR', device=device)

    def no_jersey_number(self):
        return None, 0

    def extract_numbers(self, text):
        number = ''
        for char in text:
            if char.isdigit():
                number += char
        return number if number != '' else None

    def choose_best_jersey_number(self, jersey_numbers, jn_confidences):
        if len(jersey_numbers) == 0:
            return self.no_jersey_number()
        else:
            jn_confidences = np.array(jn_confidences)
            idx_sort = np.argsort(jn_confidences)
            return jersey_numbers[idx_sort[-1]], jn_confidences[idx_sort[-1]]

    def extract_jersey_numbers_from_ocr(self, prediction):
        jersey_numbers = []
        jn_confidences = []
        for txt, conf in zip(prediction['rec_texts'], prediction['rec_scores']):
            jn = self.extract_numbers(txt)
            if jn is not None:
                jersey_numbers.append(jn)
                jn_confidences.append(conf)
        jersey_number, jn_confidence = self.choose_best_jersey_number(jersey_numbers, jn_confidences)
        if jersey_number is not None:
            jersey_number = jersey_number[:2]
        return jersey_number, jn_confidence

    @torch.no_grad()
    def process(self, batch):
        jersey_number_detection = []
        jersey_number_confidence = []
        images_np = [img.cpu().numpy() for img in batch['image']]
        del batch['image']

        predictions = self.run_mmocr_inference(images_np)
        print("Predictions:", predictions)  # Debug print

        for prediction in predictions:
            jn, conf = self.extract_jersey_numbers_from_ocr(prediction)
            jersey_number_detection.append(jn)
            jersey_number_confidence.append(conf)

        results = pd.DataFrame({
            'jersey_number_detection': jersey_number_detection,
            'jersey_number_confidence': jersey_number_confidence
        })

        return results

    def run_mmocr_inference(self, images_np):
        result = {}
        result['det'] = self.textdetinferencer(
            images_np,
            return_datasamples=True,
            batch_size=self.batch_size,
            progress_bar=False,
        )['predictions']

        print("Detection results:", result['det'])  # Debug print

        result['rec'] = []
        for img, det_data_sample in zip(images_np, result['det']):
            det_pred = det_data_sample.pred_instances
            rec_inputs = []
            for polygon in det_pred['polygons']:
                quad = bbox2poly(poly2bbox(polygon)).tolist()
                rec_input = crop_img(img, quad)
                if rec_input.shape[0] == 0 or rec_input.shape[1] == 0:
                    continue
                rec_inputs.append(rec_input)
            rec_results = self.textrecinferencer(
                rec_inputs,
                return_datasamples=True,
                batch_size=self.batch_size,
                progress_bar=False)['predictions']
            result['rec'].append(rec_results)

        print("Recognition results:", result['rec'])  # Debug print

        pred_results = [{} for _ in range(len(result['rec']))]
        for i, rec_pred in enumerate(result['rec']):
            result_out = dict(rec_texts=[], rec_scores=[])
            for rec_pred_instance in rec_pred:
                rec_dict_res = self.textrecinferencer.pred2dict(rec_pred_instance)
                result_out['rec_texts'].append(rec_dict_res['text'])
                result_out['rec_scores'].append(rec_dict_res['scores'])
            pred_results[i].update(result_out)

        return pred_results