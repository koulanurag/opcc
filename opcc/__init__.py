import os
import pickle

import d4rl
import gym
import torch

from .config import ASSETS_DIR
from .config import ENV_CONFIGS, ENV_PERFORMANCE_STATS
from .config import MAX_PRE_TRAINED_LEVEL
from .config import MIN_PRE_TRAINED_LEVEL
from .model import ActorNetwork

__all__ = [
    "get_queries",
    "get_policy",
    "get_sequence_dataset",
    "get_qlearning_dataset",
    "get_dataset_names",
    "get_env_names",
]


def get_queries(env_name):
    """
    Retrieves queries for the environment.

    :param env_name:  name of the environment
    :type env_name: str

    :return:
        A nested dictionary with the following structure:

        .. code-block:: python

            {
                (policy_a_args, policy_b_args): {
                        'obs_a': list,

                        'obs_b': list,

                        'action_a': list,

                        'action_b': list,

                        'target': list,

                        'horizon': list,
                    }

            }

    :rtype: dict

    :example:
        >>> import opcc
        >>> opcc.get_queries('Hopper-v2')

    """
    assert (
        env_name in ENV_CONFIGS
    ), "`{}` not found. It should be among following: {}".format(
        env_name, list(ENV_CONFIGS.keys())
    )

    env_dir = os.path.join(ASSETS_DIR, env_name)
    queries_path = os.path.join(env_dir, "queries.p")

    queries = pickle.load(open(queries_path, "rb"))
    return queries


def get_policy(env_name: str, pre_trained: int = 1):
    """
    Retrieves policies for the environment with the pre-trained quality marker.

    :param env_name:  name of the environment
    :type env_name: str

    :param pre_trained: pre_trained level of the policy . It should be between 1 and
                        4(inclusive) , where 1 indicates best model and 5 indicates
                        worst level.
    :type pre_trained: int

    :return:
        A tuple containing two objects: 1) policy. 2) a dictionary of performance
        stats of the policy for the given env_name

    :rtype: tuple of (ActorCriticNetwork, dict)

    :example:
        >>> import opcc, torch
        >>> policy, policy_stats = opcc.get_policy('d4rl:maze2d-open-v0',pre_trained=1)
        >>> action = policy(torch.randn(1,4).double())
        tensor([[0.9754, 0.9998]], dtype=torch.float64, grad_fn=<TanhBackward0>)
    """

    assert MIN_PRE_TRAINED_LEVEL <= pre_trained <= MAX_PRE_TRAINED_LEVEL, (
        "pre_trained marker should be between [{},{}] where {} indicates "
        "the best model and {} indicates the worst model".format(
            MIN_PRE_TRAINED_LEVEL,
            MAX_PRE_TRAINED_LEVEL,
            MIN_PRE_TRAINED_LEVEL,
            MAX_PRE_TRAINED_LEVEL,
        )
    )

    assert env_name in ENV_CONFIGS, "{} is invalid. Expected values include {}".format(
        env_name, ENV_CONFIGS.keys()
    )

    # retrieve model
    model_dir = os.path.join(ASSETS_DIR, env_name, "models")
    model_path = os.path.join(model_dir, "model_{}.p".format(pre_trained))
    assert os.path.exists(model_path), f"model not found @ {model_path}"

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    actor_state_dict = {
        k.replace("actor.", ""): v for k, v in state_dict.items() if "actor" in k
    }

    # create model
    model = ActorNetwork(
        ENV_CONFIGS[env_name]["observation_size"],
        ENV_CONFIGS[env_name]["action_size"],
        hidden_dim=64,
        max_action=ENV_CONFIGS[env_name]["max_action"],
    )
    model.load_state_dict(actor_state_dict)

    # Note: Gym returns observations with numpy float64( or double) type.
    # And, if the model is in "float" ( or float32) then we need to downcast
    # the observation to float32 before feeding them to the network.
    # However, this down-casting leads to miniscule differences in precision
    # over different system (processors). Though, these differences are
    # miniscule, they get propagated to the predicted actions which over longer
    # horizons when feedback back to the gym-environment lead to small
    # but significant difference in trajectories as reflected in monte-carlo
    # return.

    # In order to prevent above scenario, we simply upcast our model to double.
    model = model.double()
    return model, ENV_PERFORMANCE_STATS[env_name][pre_trained]


def get_sequence_dataset(env_name, dataset_name):
    """
    Retrieves episodic dataset for the given environment and dataset_name

    :param env_name:  name of the environment
    :type env_name: str

    :param dataset_name: name of the dataset
    :type dataset_name: str

    :return:
        A list of dictionaries. Each dictionary is an episode containing keys

        .. code-block:: python

            ['next_observations', 'observations', 'rewards', 'terminals', 'timeouts']`

    :rtype: list[dict]

    :example:
        >>> import opcc
        >>> dataset = opcc.get_sequence_dataset('Hopper-v2', 'medium') # list of episodes dictionaries
        >>> len(dataset)
        2186
        >>> dataset[0].keys()
        dict_keys(['actions', 'infos/action_log_probs', 'infos/qpos',
         'infos/qvel', 'next_observations', 'observations', 'rewards',
          'terminals', 'timeouts'])
        >>> len(dataset[0]['observations']) # episode length
        470
    """
    assert env_name in ENV_CONFIGS, "{} is invalid. Expected values include {}".format(
        env_name, ENV_CONFIGS.keys()
    )
    assert (
        dataset_name in ENV_CONFIGS[env_name]["datasets"]
    ), "`{}` not found. It should be among following: {}".format(
        dataset_name, list(ENV_CONFIGS[env_name]["datasets"].keys())
    )

    dataset_env = ENV_CONFIGS[env_name]["datasets"][dataset_name]["name"]
    env = gym.make(dataset_env)
    dataset = env.get_dataset()
    # remove meta-data as the sequence dataset doesn't work with it.
    metadata_keys = [k for k in dataset.keys() if "meta" in k]
    for k in metadata_keys:
        dataset.pop(k)

    split = ENV_CONFIGS[env_name]["datasets"][dataset_name]["split"]
    if split is not None:
        dataset = {k: v[:split] for k, v in dataset.items()}

    dataset = [x for x in d4rl.sequence_dataset(env, dataset)]
    return dataset


def get_qlearning_dataset(env_name, dataset_name):
    """
    Retrieves list of episodic transitions for the given environment and dataset_name

    :param env_name:  name of the environment
    :type env_name: str

    :param dataset_name: name of the dataset
    :type dataset_name: str

    :example:
        >>> import opcc
        >>> dataset = opcc.get_qlearning_dataset('Hopper-v2', 'medium') # dictionaries
        >>> dataset.keys()
        dict_keys(['observations', 'actions', 'next_observations',
         'rewards', 'terminals'])
        >>> len(dataset['observations']) # length of dataset
        999998
    """
    assert env_name in ENV_CONFIGS, "{} is invalid. Expected values include {}".format(
        env_name, ENV_CONFIGS.keys()
    )
    assert (
        dataset_name in ENV_CONFIGS[env_name]["datasets"]
    ), "`{}` not found. It should be among following: {}".format(
        dataset_name, list(ENV_CONFIGS[env_name]["datasets"].keys())
    )

    dataset_env = ENV_CONFIGS[env_name]["datasets"][dataset_name]["name"]
    env = gym.make(dataset_env)
    dataset = d4rl.qlearning_dataset(env)

    split = ENV_CONFIGS[env_name]["datasets"][dataset_name]["split"]
    if split is not None:
        dataset = {k: v[:split] for k, v in dataset.items()}
    return dataset


def get_dataset_names(env_name):
    """
    Retrieves list of dataset-names available for an environment

    :param env_name:  name of the environment
    :type env_name: str

    :return: A list of dataset-names
    :rtype: list[str]

    :example:
        >>> import opcc
        >>> opcc.get_dataset_names('Hopper-v2')
        ['random', 'expert', 'medium', 'medium-replay', 'medium-expert']
    """
    assert (
        env_name in ENV_CONFIGS
    ), "`{}` not found. It should be among following: {}".format(
        env_name, list(ENV_CONFIGS.keys())
    )
    return list(ENV_CONFIGS[env_name]["datasets"].keys())


def get_env_names():
    """
    Retrieves list of environment for which queries are available

    :return: A list of env-names
    :rtype: list[str]

    :example:
        >>> import opcc
        >>> opcc.get_env_names()
        ['HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2',
         'd4rl:maze2d-large-v1', 'd4rl:maze2d-medium-v1',
          'd4rl:maze2d-open-v0', 'd4rl:maze2d-umaze-v1']
    """
    return sorted(list(ENV_CONFIGS.keys()))
