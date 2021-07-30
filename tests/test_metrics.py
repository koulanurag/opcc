import cque
import numpy as np


def test_metric_uncertainty():
    prediction = np.ones((10, 1))
    confidence_score = 0.6 * np.ones((10, 1))
    target = np.ones((10, 1))

    accuracy, idk, info = cque.measure_uncertainty(prediction, confidence_score, target)

    assert accuracy == 1
    assert idk == 0
