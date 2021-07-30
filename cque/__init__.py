import numpy as np


def measure_uncertainty(prediction, confidence_score, target, confidence_threshold=0.5):
    match = np.zeros(prediction.shape)
    idk = np.zeros(prediction.shape)
    confident_idx = confidence_score > confidence_threshold
    match[confident_idx] = (prediction[confident_idx] == target[confident_idx]).astype(int)
    idk[~confident_idx] = 1
    return match.mean(), idk.mean(), {}
