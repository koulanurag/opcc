import os
from pathlib import Path

CQUE_DIR = os.getenv('CQUE_DIR', default=os.path.join(str(Path.home()), '.cque'))

MAZE_BASE_PROJECT_URL = 'koulanurag/cque'
MUJOCO_BASE_PROJECT_URL = 'koulanurag/cque'

MAZE_ENV_IDS = {'d4rl:maze2d-medium-v1': {'wandb_run_path': 'koulanurag/cque/1go232wj'},
                'd4rl:maze2d-open-dense-v0': {'wandb_run_path': 'koulanurag/cque/j8svyyfy'},
                'd4rl:maze2d-large-dense-v1': {'wandb_run_path': 'koulanurag/cque/f8v0novr'},
                'd4rl:maze2d-open-v0': {'wandb_run_path': 'koulanurag/cque/b9y48h5k'},
                'd4rl:maze2d-medium-dense-v1': {'wandb_run_path': 'koulanurag/cque/3ke5lwka'},
                'd4rl:maze2d-large-v1': {'wandb_run_path': 'koulanurag/cque/2tsu38l3'},
                'd4rl:maze2d-umaze-dense-v1': {'wandb_run_path': 'koulanurag/cque/1pr3nqu0'},
                'd4rl:maze2d-umaze-v1': {'wandb_run_path': 'koulanurag/cque/v0dih0r9'}}

ENV_IDS = {**MAZE_ENV_IDS}
