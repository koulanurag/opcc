import os
import pickle

import gym
import wandb
import d4rl
from .config import ENV_IDS, CQUE_DIR


def get_queries(env_name):
    assert env_name in ENV_IDS, \
        '`{}` not found. It should be among following: {}'.format(env_name, list(ENV_IDS.keys()))
    run_path = ENV_IDS[env_name]['wandb_run_path']
    env_root = os.path.join(CQUE_DIR, env_name)
    os.makedirs(env_root, exist_ok=True)
    wandb.restore(name='queries.p', run_path=run_path, replace=True, root=env_root)
    queries = pickle.load(open(os.path.join(env_root, 'queries.p'), 'rb'))
    return queries


def get_sequence_dataset(env_name, dataset_name):
    assert env_name in ENV_IDS, \
        '`{}` not found. It should be among following: {}'.format(env_name, list(ENV_IDS.keys()))
    assert dataset_name in ENV_IDS[env_name]['datasets'], \
        '`{}` not found. It should be among following: {}'.format(dataset_name,
                                                                  list(ENV_IDS[env_name]['datasets'].keys()))

    dataset_env = ENV_IDS[env_name]['datasets'][dataset_name]['name']
    env = gym.make(dataset_env)
    dataset = env.get_dataset()
    # remove meta-data as the sequence dataset doesn't work with it.
    metadata_keys = [k for k in dataset.keys() if 'meta' in k]
    for k in metadata_keys:
        dataset.pop(k)

    split = ENV_IDS[env_name]['datasets'][dataset_name]['split']
    if split is not None:
        dataset = {k: v[:split] for k, v in dataset.items()}

    dataset = [x for x in d4rl.sequence_dataset(env, dataset)]
    return dataset


def get_dataset_names(env_name):
    assert env_name in ENV_IDS, \
        '`{}` not found. It should be among following: {}'.format(env_name, list(ENV_IDS.keys()))
    return list(ENV_IDS[env_name]['datasets'].keys())
