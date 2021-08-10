import numpy as np
from .config import ENV_IDS, CQUE_DIR
import wandb
import os
import pickle


def measure_uncertainty(prediction, uncertainty_score, target, uncertainty_threshold):
    match = np.zeros(prediction.shape)
    abstain = np.zeros(prediction.shape)
    certain_idx = uncertainty_score <= uncertainty_threshold
    match[certain_idx] = (prediction[certain_idx] == target[certain_idx]).astype(int)
    abstain[~certain_idx] = 1
    return match.mean() if sum(match) != 0 else 1, abstain.mean(), {}


def get_queries(env_name):
    assert env_name in ENV_IDS, \
        '`{}` not found. It should be among following: {}'.format(env_name, list(ENV_IDS.keys()))
    run_path = ENV_IDS[env_name]['wandb_run_path']
    env_root = os.path.join(CQUE_DIR, env_name)
    os.makedirs(env_root, exist_ok=True)
    wandb.restore(name='queries.p', run_path=run_path, replace=True, root=env_root)
    queries = pickle.load(open(os.path.join(env_root, 'queries.p'), 'rb'))
    return queries
