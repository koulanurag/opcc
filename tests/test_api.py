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

    for (policy_a_id, policy_b_id), query_batch in queries.items():
        policy_a = policybazaar.get_policy(*policy_a_id)
        policy_b = policybazaar.get_policy(*policy_b_id)

        keys = ['obs_a', 'obs_a', 'action_a', 'action_b', 'horizon', 'target', 'info']
        for key in keys:
            assert key in query_batch, '{} not in query_batch'.format(key)

        info_keys = ['return_a', 'return_b', 'state_a', 'state_b', 'runs', 'horizon_a', 'horizon_b']
        for key in keys:
            assert key in query_batch['info'], '{} not in query_batch'.format(key)

        assert np.mean([len(query_batch[key] for key in keys)] + [len(query_batch['info'][key] for key in info_keys)]) \
               == len(query_batch['oba_a']), ' batch sizes doesn\'t match'


@pytest.mark.parametrize('env_name,dataset_name', DATASET_ENV_PAIRS)
def test_get_dataset(env_name, dataset_name):
    dataset = cque.get_dataset(env_name, dataset_name)
