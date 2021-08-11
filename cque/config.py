import os
from pathlib import Path

CQUE_DIR = os.getenv('CQUE_DIR', default=os.path.join(str(Path.home()), '.cque'))

MAZE_BASE_PROJECT_URL = 'koulanurag/cque'
MUJOCO_BASE_PROJECT_URL = 'koulanurag/cque'

MAZE_ENV_IDS = {'d4rl:maze2d-medium-v1': {'wandb_run_path': 'koulanurag/cque/awk23x2d'},
                'd4rl:maze2d-umaze-v1': {'wandb_run_path': 'koulanurag/cque/3vv1qbbu'},
                'd4rl:maze2d-umaze-dense-v1': {'wandb_run_path': 'koulanurag/cque/3tr3yqx8'},
                'd4rl:maze2d-large-v1': {'wandb_run_path': 'koulanurag/cque/3oqi1usx'},
                'd4rl:maze2d-open-dense-v0': {'wandb_run_path': 'koulanurag/cque/30i0aoyh'},
                'd4rl:maze2d-medium-dense-v1': {'wandb_run_path': 'koulanurag/cque/2nxd6dzp'},
                'd4rl:maze2d-open-v0': {'wandb_run_path': 'koulanurag/cque/2akwjbh0'},
                'd4rl:maze2d-large-dense-v1': {'wandb_run_path': 'koulanurag/cque/1lh1snup'}}

ENV_IDS = {**MAZE_ENV_IDS}
