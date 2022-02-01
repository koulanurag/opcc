import os
import pickle

import d4rl
import gym
import torch
from .config import ENV_CONFIGS, ENV_PERFORMANCE_STATS
from pathlib import Path

from .config import MIN_PRE_TRAINED_LEVEL
from .config import MAX_PRE_TRAINED_LEVEL


def get_queries(env_name):
    """
    Retrieves queries for the environment.

    :param env_name:  name of the environment

    Example:
        >>> import opcc
        >>> opcc.get_queries('Hopper-v2')
    """
    assert env_name in ENV_CONFIGS, \
        ('`{}` not found. It should be among following: {}'
         .format(env_name, list(ENV_CONFIGS.keys())))

    env_dir = os.path.join(Path(os.path.dirname(__file__)).parent,
                           'assets', env_name)
    queries_path = os.path.join(env_dir, 'queries.p')

    queries = pickle.load(open(queries_path, 'rb'))
    return queries


def get_policy(env_name: str, pre_trained: int = 1):
    """
    Retrieves policies for the environment with the pre-trained quality marker.

    :param env_name:  name of the environment
    :param pre_trained: pre_trained level . It should be between 1 and 5 ,
                        where 1 indicates best model and 5 indicates worst
                        level.

    Example:
        >>> import opcc
        >>> opcc.get_policy('d4rl:maze2d-open-v0',pre_trained=1)
    """

    assert MIN_PRE_TRAINED_LEVEL <= pre_trained <= MAX_PRE_TRAINED_LEVEL, \
        ('pre_trained marker should be between [{},{}] where {} indicates '
         'the best model and {} indicates the worst model'
         .format(MIN_PRE_TRAINED_LEVEL, MAX_PRE_TRAINED_LEVEL,
                 MIN_PRE_TRAINED_LEVEL, MAX_PRE_TRAINED_LEVEL))

    assert env_name in ENV_CONFIGS, \
        ('{} is invalid. Expected values include {}'
         .format(env_name, ENV_CONFIGS.keys()))

    # retrieve model
    model_dir = os.path.join(Path(os.path.dirname(__file__)).parent, 'assets',
                             env_name, 'models')
    model_path = os.path.join(model_dir, 'model_{}.p'.format(pre_trained))
    assert os.path.exists(model_path), \
        'model not found @ {}'.format(model_path)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    # create model
    from .model import ActorCriticNetwork
    model = ActorCriticNetwork(ENV_CONFIGS[env_name]['observation_size'],
                               ENV_CONFIGS[env_name]['action_size'],
                               hidden_dim=64,
                               action_std=0.5)
    model.load_state_dict(state_dict)

    return model, ENV_PERFORMANCE_STATS[env_name][pre_trained]


def get_sequence_dataset(env_name, dataset_name):
    assert env_name in ENV_CONFIGS, \
        ('{} is invalid. Expected values include {}'
         .format(env_name, ENV_CONFIGS.keys()))
    assert dataset_name in ENV_CONFIGS[env_name]['datasets'], \
        ('`{}` not found. It should be among following: {}'.
         format(dataset_name, list(ENV_CONFIGS[env_name]['datasets'].keys())))

    dataset_env = ENV_CONFIGS[env_name]['datasets'][dataset_name]['name']
    env = gym.make(dataset_env)
    dataset = env.get_dataset()
    # remove meta-data as the sequence dataset doesn't work with it.
    metadata_keys = [k for k in dataset.keys() if 'meta' in k]
    for k in metadata_keys:
        dataset.pop(k)

    split = ENV_CONFIGS[env_name]['datasets'][dataset_name]['split']
    if split is not None:
        dataset = {k: v[:split] for k, v in dataset.items()}

    dataset = [x for x in d4rl.sequence_dataset(env, dataset)]
    return dataset


def get_qlearning_dataset(env_name, dataset_name):
    assert env_name in ENV_CONFIGS, \
        ('{} is invalid. Expected values include {}'
         .format(env_name, ENV_CONFIGS.keys()))
    assert dataset_name in ENV_CONFIGS[env_name]['datasets'], \
        ('`{}` not found. It should be among following: {}'.
         format(dataset_name, list(ENV_CONFIGS[env_name]['datasets'].keys())))

    dataset_env = ENV_CONFIGS[env_name]['datasets'][dataset_name]['name']
    env = gym.make(dataset_env)
    dataset = d4rl.qlearning_dataset(env)

    split = ENV_CONFIGS[env_name]['datasets'][dataset_name]['split']
    if split is not None:
        dataset = {k: v[:split] for k, v in dataset.items()}
    return dataset


def get_dataset_names(env_name):
    assert env_name in ENV_CONFIGS, \
        ('`{}` not found. It should be among following: {}'.
         format(env_name, list(ENV_CONFIGS.keys())))
    return list(ENV_CONFIGS[env_name]['datasets'].keys())
