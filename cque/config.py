import os
from pathlib import Path

CQUE_DIR = os.getenv('CQUE_DIR', default=os.path.join(str(Path.home()), '.cque'))

MAZE_BASE_PROJECT_URL = 'koulanurag/cque'
MUJOCO_BASE_PROJECT_URL = 'koulanurag/cque'

MAZE_ENV_IDS = {'d4rl:maze2d-large-v1': {'wandb_run_path': 'koulanurag/cque/z9gb1mn1',
                                         'datasets': {'1k': {'name': 'd4rl:maze2d-large-v1', 'split': 1000},
                                                      '10k': {'name': 'd4rl:maze2d-large-v1', 'split': 10000},
                                                      '100k': {'name': 'd4rl:maze2d-large-v1', 'split': 100000}}},
                'd4rl:maze2d-umaze-v1': {'wandb_run_path': 'koulanurag/cque/2m837r2f',
                                         'datasets': {'1k': {'name': 'd4rl:maze2d-umaze-v1', 'split': 1000},
                                                      '10k': {'name': 'd4rl:maze2d-umaze-v1', 'split': 10000},
                                                      '100k': {'name': 'd4rl:maze2d-umaze-v1', 'split': 100000}}},
                'd4rl:maze2d-open-v0': {'wandb_run_path': 'koulanurag/cque/u0xm82cc',
                                        'datasets': {'1k': {'name': 'd4rl:maze2d-open-v0', 'split': 1000},
                                                     '10k': {'name': 'd4rl:maze2d-open-v0', 'split': 10000},
                                                     '100k': {'name': 'd4rl:maze2d-open-v0', 'split': 100000}}},
                'd4rl:maze2d-medium-v1': {'wandb_run_path': 'koulanurag/cque/15rze6rj',
                                          'datasets': {'1k': {'name': 'd4rl:maze2d-medium-v1', 'split': 1000},
                                                       '10k': {'name': 'd4rl:maze2d-medium-v1', 'split': 10000},
                                                       '100k': {'name': 'd4rl:maze2d-medium-v1', 'split': 100000}}}}

MUJOCO_ENV_IDS = {'HalfCheetah-v2': {'wandb_run_path': 'koulanurag/cque/285ea0rt',
                                     'datasets': {'random': {'name': 'd4rl:halfcheetah-random-v2', 'split': None},
                                                  'expert': {'name': 'd4rl:halfcheetah-expert-v2', 'split': None},
                                                  'medium': {'name': 'd4rl:halfcheetah-medium-v2', 'split': None},
                                                  'medium-replay': {'name': 'd4rl:halfcheetah-medium-replay-v2',
                                                                    'split': None},
                                                  'medium-expert': {'name': 'd4rl:halfcheetah-medium-expert-v2',
                                                                    'split': None}}},
                  'Walker2d-v2': {'wandb_run_path': 'koulanurag/cque/9b3uud9p',
                                  'datasets': {'random': {'name': 'd4rl:walker2d-random-v2', 'split': None},
                                               'expert': {'name': 'd4rl:walker2d-expert-v2', 'split': None},
                                               'medium': {'name': 'd4rl:walker2d-medium-v2', 'split': None},
                                               'medium-replay': {'name': 'd4rl:walker2d-medium-replay-v2',
                                                                 'split': None},
                                               'medium-expert': {'name': 'd4rl:walker2d-medium-expert-v2',
                                                                 'split': None}}},
                  'Hopper-v2': {'wandb_run_path': 'koulanurag/cque/3kuyxvyg',
                                'datasets': {'random': {'name': 'd4rl:hopper-random-v2', 'split': None},
                                             'expert': {'name': 'd4rl:hopper-expert-v2', 'split': None},
                                             'medium': {'name': 'd4rl:hopper-medium-v2', 'split': None},
                                             'medium-replay': {'name': 'd4rl:hopper-medium-replay-v2', 'split': None},
                                             'medium-expert': {'name': 'd4rl:hopper-medium-expert-v2', 'split': None}}}}

ENV_IDS = {**MAZE_ENV_IDS, **MUJOCO_ENV_IDS}
