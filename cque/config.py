import os
from pathlib import Path

CQUE_DIR = os.getenv('CQUE_DIR', default=os.path.join(str(Path.home()), '.cque'))

MAZE_BASE_PROJECT_URL = 'koulanurag/cque'
MUJOCO_BASE_PROJECT_URL = 'koulanurag/cque'

MAZE_ENV_IDS = {'d4rl:maze2d-large-v1': {'wandb_run_path': 'koulanurag/cque/2jpytkj9'},
                'd4rl:maze2d-medium-dense-v1': {'wandb_run_path': 'koulanurag/cque/21f6capa'},
                'd4rl:maze2d-medium-v1': {'wandb_run_path': 'koulanurag/cque/57zmcmti'},
                'd4rl:maze2d-umaze-dense-v1': {'wandb_run_path': 'koulanurag/cque/3oz7iz95'},
                'd4rl:maze2d-large-dense-v1': {'wandb_run_path': 'koulanurag/cque/2sa8v52m'},
                'd4rl:maze2d-open-v0': {'wandb_run_path': 'koulanurag/cque/2ryjyty0'},
                'd4rl:maze2d-umaze-v1': {'wandb_run_path': 'koulanurag/cque/2cg2at9k'},
                'd4rl:maze2d-open-dense-v0': {'wandb_run_path': 'koulanurag/cque/1vb4o2fl'}}

ENV_IDS = {**MAZE_ENV_IDS}
