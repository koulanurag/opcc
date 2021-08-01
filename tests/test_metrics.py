import cque
import numpy as np
import policybazaar


def test_metric_uncertainty():
    prediction = np.ones((10, 1))
    confidence_score = 0.6 * np.ones((10, 1))
    target = np.ones((10, 1))

    accuracy, idk, info = cque.measure_uncertainty(prediction, confidence_score, target)

    assert accuracy == 1
    assert idk == 0


def test_get_queries(env_name):
    queries = cque.get_queries(env_name)

    for (policy_a_id, policy_b_id) in queries:
        policy_a = policybazaar.get_policy(policy_a_id)
        policy_b = policybazaar.get_policy(policy_b_id)

        state_a, action_a, state_b, action_b, target = queries[(policy_a_id, policy_b_id)]
