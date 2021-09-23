import os
from pathlib import Path

CQUE_DIR = os.getenv('CQUE_DIR', default=os.path.join(str(Path.home()), '.cque'))

MAZE_BASE_PROJECT_URL = 'koulanurag/cque'
MUJOCO_BASE_PROJECT_URL = 'koulanurag/cque'

MAZE_ENV_IDS = {'d4rl:walker2d-random-v2': {'wandb_run_path': 'koulanurag/cque/39l783xs'},
                'd4rl:hopper-random-v2': {'wandb_run_path': 'koulanurag/cque/377q6u5r'},
                'd4rl:halfcheetah-random-v2': {'wandb_run_path': 'koulanurag/cque/28t69ki6'},
                'd4rl:maze2d-umaze-dense-v1': {'wandb_run_path': 'koulanurag/cque/a72gmlma'},
                'd4rl:maze2d-open-dense-v0': {'wandb_run_path': 'koulanurag/cque/2mg1cey3'},
                'd4rl:maze2d-large-v1': {'wandb_run_path': 'koulanurag/cque/2dy4i23z'},
                'd4rl:maze2d-umaze-v1': {'wandb_run_path': 'koulanurag/cque/1gvdzev8'},
                'd4rl:maze2d-medium-v1': {'wandb_run_path': 'koulanurag/cque/fh34w5xq'},
                'd4rl:maze2d-open-v0': {'wandb_run_path': 'koulanurag/cque/2fw51oua'},
                'd4rl:maze2d-medium-dense-v1': {'wandb_run_path': 'koulanurag/cque/21qelw3l'},
                'd4rl:maze2d-large-dense-v1': {'wandb_run_path': 'koulanurag/cque/1ziou54w'},
                'd4rl:halfcheetah-expert-v2': {'wandb_run_path': 'koulanurag/cque/28t69ki6'},
                'd4rl:halfcheetah-medium-v2': {'wandb_run_path': 'koulanurag/cque/28t69ki6'},
                'd4rl:halfcheetah-medium-replay-v2': {'wandb_run_path': 'koulanurag/cque/28t69ki6'},
                'd4rl:halfcheetah-medium-expert-v2': {'wandb_run_path': 'koulanurag/cque/28t69ki6'},
                'd4rl:halfcheetah-expert-v2': {'wandb_run_path': 'koulanurag/cque/28t69ki6'},
                'd4rl:walker2d-expert-v2': {'wandb_run_path': 'koulanurag/cque/39l783xs'},
                'd4rl:walker2d-medium-v2': {'wandb_run_path': 'koulanurag/cque/39l783xs'},
                'd4rl:walker2d-medium-replay-v2': {'wandb_run_path': 'koulanurag/cque/39l783xs'},
                'd4rl:walker2d-medium-expert-v2': {'wandb_run_path': 'koulanurag/cque/39l783xs'},
                'd4rl:hopper-expert-v2': {'wandb_run_path': 'koulanurag/cque/377q6u5r'},
                'd4rl:hopper-medium-v2': {'wandb_run_path': 'koulanurag/cque/377q6u5r'},
                'd4rl:hopper-medium-replay-v2': {'wandb_run_path': 'koulanurag/cque/377q6u5r'},
                'd4rl:hopper-medium-expert-v2': {'wandb_run_path': 'koulanurag/cque/377q6u5r'}}

ENV_IDS = {**MAZE_ENV_IDS}
