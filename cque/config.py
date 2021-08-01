import os
from pathlib import Path

CQUE_DIR = os.getenv('CQUE_DIR', default=os.path.join(str(Path.home()), '.cque'))

MAZE_BASE_PROJECT_URL = 'koulanurag/cque'
MUJOCO_BASE_PROJECT_URL = 'koulanurag/cque'

MAZE_ENV_IDS = {'d4rl:maze2d-large-v1': {'wandb_run_path': 'koulanurag/cque/zo1lr2yl'},
                'd4rl:maze2d-umaze-dense-v1': {'wandb_run_path': 'koulanurag/cque/g5ickox6'},
                'd4rl:maze2d-large-dense-v1': {'wandb_run_path': 'koulanurag/cque/3vpt5b85'},
                'd4rl:maze2d-medium-v1': {'wandb_run_path': 'koulanurag/cque/3leq39zm'},
                'd4rl:maze2d-open-v0': {'wandb_run_path': 'koulanurag/cque/3cw56ib3'},
                'd4rl:maze2d-open-dense-v0': {'wandb_run_path': 'koulanurag/cque/311n91at'},
                'd4rl:maze2d-medium-dense-v1': {'wandb_run_path': 'koulanurag/cque/21ftmgzh'},
                'd4rl:maze2d-umaze-v1': {'wandb_run_path': 'koulanurag/cque/1xh3iiyq'}}
ENV_IDS = {**MAZE_ENV_IDS}
