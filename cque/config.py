import os
from pathlib import Path

CQUE_DIR = os.getenv('CQUE_DIR', default=os.path.join(str(Path.home()), '.cque'))

MAZE_BASE_PROJECT_URL = 'koulanurag/cque'
MUJOCO_BASE_PROJECT_URL = 'koulanurag/cque'

MAZE_ENV_IDS = {}

MUJOCO_ENV_IDS = {'Hopper-v2': {'wandb_run_path': 'koulanurag/cque/2qydkdze',
                                'datasets': {'random': {'name': 'd4rl:hopper-random-v2', 'split': None},
                                             'expert': {'name': 'd4rl:hopper-expert-v2', 'split': None},
                                             'medium': {'name': 'd4rl:hopper-medium-v2', 'split': None},
                                             'medium-replay': {'name': 'd4rl:hopper-medium-replay-v2', 'split': None},
                                             'medium-expert': {'name': 'd4rl:hopper-medium-expert-v2', 'split': None}}},
                  'Walker2d-v2': {'wandb_run_path': 'koulanurag/cque/7ggorwx5',
                                  'datasets': {'random': {'name': 'd4rl:walker2d-random-v2', 'split': None},
                                               'expert': {'name': 'd4rl:walker2d-expert-v2', 'split': None},
                                               'medium': {'name': 'd4rl:walker2d-medium-v2', 'split': None},
                                               'medium-replay': {'name': 'd4rl:walker2d-medium-replay-v2',
                                                                 'split': None},
                                               'medium-expert': {'name': 'd4rl:walker2d-medium-expert-v2',
                                                                 'split': None}}},
                  'HalfCheetah-v2': {'wandb_run_path': 'koulanurag/cque/1x84ljtq',
                                     'datasets': {'random': {'name': 'd4rl:halfcheetah-random-v2', 'split': None},
                                                  'expert': {'name': 'd4rl:halfcheetah-expert-v2', 'split': None},
                                                  'medium': {'name': 'd4rl:halfcheetah-medium-v2', 'split': None},
                                                  'medium-replay': {'name': 'd4rl:halfcheetah-medium-replay-v2',
                                                                    'split': None},
                                                  'medium-expert': {'name': 'd4rl:halfcheetah-medium-expert-v2',
                                                                    'split': None}}}}

ENV_IDS = {**MAZE_ENV_IDS, **MUJOCO_ENV_IDS}
