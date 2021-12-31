import gym
import numpy as np
import policybazaar
import pytest
import torch
from gym.vector.sync_vector_env import SyncVectorEnv

import cque
from cque.config import ENV_IDS

DATASET_ENV_PAIRS = []
for _env_name in ENV_IDS.keys():
    DATASET_ENV_PAIRS += [(_env_name, dataset_name)
                          for dataset_name in ENV_IDS[_env_name]['datasets']]


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
        for key in info_keys:
            assert key in query_batch['info'], '{} not in query_batch'.format(key)

        avg_batch_size = np.mean([len(query_batch[key]) for key in keys if key != 'info']
                                 + [len(query_batch['info'][key]) for key in info_keys if key != 'runs'])
        assert avg_batch_size == len(query_batch['obs_a']), ' batch sizes does not match'


def mc_return(env_name, state_a, init_obs, init_action, policy, horizon: int,
              device='cpu', runs=1, step_batch_size=128):
    batch_size, obs_size = init_obs.shape
    _, action_size = init_action.shape

    with torch.no_grad():
        # setup initial obs and action
        step_obs = torch.FloatTensor(init_obs).to(device)
        init_action = torch.FloatTensor(init_action).to(device)
        step_obs = step_obs.repeat(runs, 1)
        init_action = init_action.repeat(runs, 1)

        # setup env
        envs = SyncVectorEnv([lambda: gym.make(env_name)
                              for _ in range(len(state_a) * runs)])
        envs.reset()
        for i, env in enumerate(envs.envs):
            env.sim.set_state_from_flattened(state_a[i % len(state_a)])
            env.sim.forward()

        # rollout
        returns = np.zeros((batch_size * runs))
        dones = np.zeros((batch_size * runs), dtype=bool)
        for step in range(horizon):
            for batch_idx in range(0, returns.shape[0], step_batch_size):
                if step == 0:
                    step_action = init_action[batch_idx:batch_idx + step_batch_size].to(device)
                else:
                    _step_obs = step_obs[batch_idx: batch_idx + step_batch_size].to(device)
                    step_action = policy.actor(_step_obs)
                step_action = step_action.cpu().numpy()

                next_obs, reward, done = [], [], []
                for env_i, env in enumerate(envs.envs[batch_idx: batch_idx + step_batch_size]):
                    _next_obs, _reward, _done, info = env.step(step_action[env_i])
                    next_obs.append(_next_obs)
                    reward.append(_reward)
                    done.append(_done)
                reward = np.array(reward)
                not_done_filter = ~dones[batch_idx: batch_idx + step_batch_size]
                returns[batch_idx: batch_idx + step_batch_size][not_done_filter] += reward[not_done_filter]
                dones[batch_idx:batch_idx + step_batch_size] = done

                step_obs[batch_idx: batch_idx + step_batch_size] = torch.Tensor(next_obs).to(device)

    returns = returns.reshape((runs, batch_size))
    returns = returns.mean(0)
    envs.close()
    return returns


@pytest.mark.parametrize('env_name', ENV_IDS.keys())
def test_query_targets(env_name):
    queries = cque.get_queries(env_name)

    for (policy_a_id, policy_b_id), query_batch in queries.items():
        policy_a, _ = policybazaar.get_policy(*policy_a_id)
        policy_b, _ = policybazaar.get_policy(*policy_b_id)
        target = query_batch['target']
        horizons = query_batch['horizon']

        for horizon in np.unique(horizons, return_counts=False):
            _filter = horizons == horizon
            state_a = query_batch['info']['state_a'][_filter]
            state_b = query_batch['info']['state_b'][_filter]
            obs_a = query_batch['obs_a'][_filter]
            obs_b = query_batch['obs_b'][_filter]
            action_a = query_batch['action_a'][_filter]
            action_b = query_batch['action_b'][_filter]
            return_a = mc_return(env_name, state_a, obs_a, action_a, policy_a, horizon,
                                 runs=query_batch['info']['runs'])
            return_b = mc_return(env_name, state_b, obs_b, action_b, policy_b, horizon,
                                 runs=query_batch['info']['runs'])
            predict = return_a < return_b
            assert all(target[_filter] == predict), 'Query targets do not match'


@pytest.mark.parametrize('env_name,dataset_name', DATASET_ENV_PAIRS)
def test_get_dataset(env_name, dataset_name):
    dataset = cque.get_dataset(env_name, dataset_name)
