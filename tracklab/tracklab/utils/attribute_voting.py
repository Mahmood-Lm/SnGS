
def select_highest_voted_att(atts, atts_confidences=None):
        
    confidence_sum = {}
    atts_confidences = [1] * len(atts) if atts_confidences is None else atts_confidences
    
    # Iterate through the predictions to calculate the total confidence for each attribute
    for jn, conf in zip(atts, atts_confidences):
        if jn not in confidence_sum:
            confidence_sum[jn] = 0
        confidence_sum[jn] += conf
    
    # Find the attribute with the maximum total confidence
    if len(confidence_sum) == 0:
        return None
    max_confidence_att = max(confidence_sum, key=confidence_sum.get)
    return max_confidence_att


def select_heaviest_voted_att(atts, atts_confidences=None, min_confidence=0.0, confidence_weight=2.0):  #FIXME
    # Confidence weight is used to adjust the importance of the confidence value. set to 1.0 to keep the confidence as is. set to 0.0 to ignore the confidence.
    confidence_sum = {}
    count_sum = {}
    atts_confidences = [1] * len(atts) if atts_confidences is None else atts_confidences
    
    # Iterate through the predictions to calculate the total confidence for each attribute
    for att, conf in zip(atts, atts_confidences):
        if conf < min_confidence:
            continue
        if att not in confidence_sum:
            confidence_sum[att] = 0
            count_sum[att] = 0
        confidence_sum[att] += conf ** confidence_weight  # Apply the confidence weight
        count_sum[att] += 1
    
    # Calculate weighted average confidence
    weighted_avg_confidence = {att: confidence_sum[att] / count_sum[att] for att in confidence_sum}
    
    # Find the attribute with the maximum weighted average confidence
    if len(weighted_avg_confidence) == 0:
        return None
    max_confidence_att = max(weighted_avg_confidence, key=weighted_avg_confidence.get)
    return max_confidence_att