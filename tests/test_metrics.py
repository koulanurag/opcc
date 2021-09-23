import cque
import numpy as np
import policybazaar
import pytest
from cque.config import ENV_IDS


def test_metric_uncertainty():
    prediction = np.ones((10, 1))
    confidence_score = 0.6 * np.ones((10, 1))
    target = np.ones((10, 1))

    accuracy, idk, info = cque.measure_uncertainty(prediction, confidence_score, target)

    assert accuracy == 1
    assert idk == 0


@pytest.mark.parametrize('env_name', ENV_IDS.keys())
def test_get_queries(env_name):
    queries = cque.get_queries(env_name)

    for (policy_a_id, policy_b_id) in queries:
        env_name_a, pre_trained_id_a = policy_a_id
        env_name_b, pre_trained_id_b = policy_b_id
        policy_a = policybazaar.get_policy(env_name_a, pre_trained_id_a)
        policy_b = policybazaar.get_policy(env_name_b, pre_trained_id_b)

        state_a, action_a, state_b, action_b, target_a, target_b, target = queries[(policy_a_id, policy_b_id)]

@pytest.mark.parametrize('dataset_name', ENV_IDS.keys())
def test_get_dataset(dataset_name):
    dataset = cque.get_dataset(dataset_name)