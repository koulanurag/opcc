import gym
import numpy as np
import pytest
import torch

import opcc
from opcc.config import ENV_CONFIGS

DATASET_ENV_PAIRS = []
for _env_name in ENV_CONFIGS.keys():
    DATASET_ENV_PAIRS += [(_env_name, dataset_name)
                          for dataset_name in
                          ENV_CONFIGS[_env_name]['datasets']]


def mc_return(env_name, sim_states, init_actions, horizon, policy, runs):
    all_returns, all_steps = [], []
    for sim_state_i, sim_state in enumerate(sim_states):
        envs = []
        for _ in range(runs):
            env = gym.make(env_name)
            env.reset()
            env.sim.set_state_from_flattened(sim_state)
            env.sim.forward()
            envs.append(env)

        obss = [None for _ in range(runs)]
        dones = [False for _ in range(runs)]
        returns = [0 for _ in range(runs)]
        steps = [0 for _ in range(runs)]

        for step_count in range(horizon):
            for env_i, env in enumerate(envs):
                if not dones[env_i]:
                    if step_count == 0:
                        obs, reward, done, info = env.step(
                            init_actions[sim_state_i])
                    else:
                        with torch.no_grad():
                            obs = torch.tensor(obss[env_i]).unsqueeze(
                                0).float()
                            action = policy.actor(obs).data.cpu().numpy()[0]
                        obs, reward, done, info = env.step(action)
                    obss[env_i] = obs
                    dones[env_i] = done or dones[env_i]
                    returns[env_i] += reward
                    steps[env_i] += 1

        [env.close() for env in envs]
        all_returns.append(returns)
        all_steps.append(steps)

    return np.array(all_returns).mean(1)


@pytest.mark.parametrize('env_name', ENV_CONFIGS.keys())
def test_get_queries(env_name):
    keys = ['obs_a', 'obs_a', 'action_a', 'action_b',
            'horizon', 'target', 'info']
    info_keys = ['return_a', 'return_b', 'state_a', 'state_b',
                 'runs', 'horizon_a', 'horizon_b']

    queries = opcc.get_queries(env_name)
    for (policy_a_id, policy_b_id), query_batch in queries.items():
        policy_a = opcc.get_policy(*policy_a_id)
        policy_b = opcc.get_policy(*policy_b_id)

        for key in keys:
            assert key in query_batch, '{} not in query_batch'.format(key)

        for key in info_keys:
            assert key in query_batch['info'], \
                '{} not in query_batch'.format(key)

        avg_batch_size = np.mean([len(query_batch[key])
                                  for key in keys if key != 'info']
                                 + [len(query_batch['info'][key])
                                    for key in info_keys if key != 'runs'])
        assert avg_batch_size == len(query_batch['obs_a']), \
            ' batch sizes does not match'


@pytest.mark.parametrize('env_name', ENV_CONFIGS.keys())
def test_query_targets(env_name):
    queries = opcc.get_queries(env_name)

    for (policy_a_id, policy_b_id), query_batch in queries.items():
        policy_a, _ = opcc.get_policy(*policy_a_id)
        policy_b, _ = opcc.get_policy(*policy_b_id)
        target = query_batch['target']
        horizons = query_batch['horizon']

        for horizon in np.unique(horizons, return_counts=False):
            _filter = horizons == horizon
            state_a = query_batch['info']['state_a'][_filter]
            state_b = query_batch['info']['state_b'][_filter]
            action_a = query_batch['action_a'][_filter]
            action_b = query_batch['action_b'][_filter]
            return_a = mc_return(env_name, state_a, action_a, horizon,
                                 policy_a, query_batch['info']['runs'])
            return_b = mc_return(env_name, state_b, action_b, horizon,
                                 policy_b, query_batch['info']['runs'])
            predict = return_a < return_b
            assert all(target[_filter] == predict), \
                'Query targets do not match for ' \
                'policies: {} and horizon: {}'.format((policy_a_id,
                                                       policy_b_id), horizon)


@pytest.mark.parametrize('env_name,dataset_name', DATASET_ENV_PAIRS)
def test_get_qlearning_dataset(env_name, dataset_name):
    dataset = opcc.get_qlearning_dataset(env_name, dataset_name)


@pytest.mark.parametrize('env_name,dataset_name', DATASET_ENV_PAIRS)
def test_get_sequence_dataset(env_name, dataset_name):
    dataset = opcc.get_sequence_dataset(env_name, dataset_name)
