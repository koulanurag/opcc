import cque
import policybazaar
import pytest
from cque.config import ENV_IDS

DATASET_ENV_PAIRS = []
for env_name in ENV_IDS.keys():
    DATASET_ENV_PAIRS += [(env_name, dataset_name)
                          for dataset_name in ENV_IDS[env_name]['datasets']]


@pytest.mark.parametrize('env_name', ENV_IDS.keys())
def test_get_queries(env_name):
    queries = cque.get_queries(env_name)

    for (policy_a_id, policy_b_id), query_batch in enumerate(queries):
        policy_a = policybazaar.get_policy(**policy_a_id)
        policy_b = policybazaar.get_policy(**policy_b_id)

        state_a, action_a, state_b, action_b, target_a, target_b, target = query_batch


@pytest.mark.parametrize('env_name,dataset_name', DATASET_ENV_PAIRS)
def test_get_dataset(env_name, dataset_name):
    dataset = cque.get_dataset(env_name, dataset_name)
