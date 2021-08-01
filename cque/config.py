import os
from pathlib import Path

CQUE_DIR = os.getenv('CQUE_DIR', default=os.path.join(str(Path.home()), '.cque'))

MAZE_BASE_PROJECT_URL = 'koulanurag/cque/'
MUJOCO_BASE_PROJECT_URL = 'koulanurag/cque/'

MAZE_ENV_IDS = {
    # sparse
    'd4rl:maze2d-open-v0': {'wandb_run_path': MAZE_BASE_PROJECT_URL + '3rd5zlnh'},
    'd4rl:maze2d-medium-v1': {'wandb_run_path': MAZE_BASE_PROJECT_URL + 'gplzxs8a'},
    'd4rl:maze2d-umaze-v1': {'wandb_run_path': MAZE_BASE_PROJECT_URL + 'e15ildhv'},
    'd4rl:maze2d-large-v1': {'wandb_run_path': MAZE_BASE_PROJECT_URL + '1vi4gkf8'},

    # sparse
    'd4rl:maze2d-open-v0': {'wandb_run_path': MAZE_BASE_PROJECT_URL + '3rd5zlnh'},
    'd4rl:maze2d-medium-v1': {'wandb_run_path': MAZE_BASE_PROJECT_URL + 'gplzxs8a'},
    'd4rl:maze2d-umaze-v1': {'wandb_run_path': MAZE_BASE_PROJECT_URL + 'e15ildhv'},
    'd4rl:maze2d-large-v1': {'wandb_run_path': MAZE_BASE_PROJECT_URL + '1vi4gkf8'},
}

ENV_IDS = {**MAZE_ENV_IDS}
