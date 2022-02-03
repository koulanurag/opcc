MAZE_ENV_CONFIGS = \
    {'d4rl:maze2d-umaze-v1':
         {'observation_size': 4,
          'action_size': 2,
          'datasets': {'1k': {'name': 'd4rl:maze2d-umaze-v1',
                              'split': 1000},
                       '10k': {'name': 'd4rl:maze2d-umaze-v1',
                               'split': 10000},
                       '100k': {'name': 'd4rl:maze2d-umaze-v1',
                                'split': 100000},
                       '1m': {'name': 'd4rl:maze2d-umaze-v1',
                              'split': 1000000},
                       }
          },
     'd4rl:maze2d-medium-v1':
         {'observation_size': 4,
          'action_size': 2,
          'datasets': {'1k': {'name': 'd4rl:maze2d-medium-v1',
                              'split': 1000},
                       '10k': {'name': 'd4rl:maze2d-medium-v1',
                               'split': 10000},
                       '100k': {'name': 'd4rl:maze2d-medium-v1',
                                'split': 100000},
                       '1m': {'name': 'd4rl:maze2d-medium-v1',
                              'split': 1000000}}
          },
     'd4rl:maze2d-large-v1':
         {'observation_size': 4,
          'action_size': 2,
          'datasets': {
              '1k': {'name': 'd4rl:maze2d-large-v1',
                     'split': 1000},
              '10k': {'name': 'd4rl:maze2d-large-v1',
                      'split': 10000},
              '100k': {'name': 'd4rl:maze2d-large-v1',
                       'split': 100000},
              '1m': {'name': 'd4rl:maze2d-large-v1',
                     'split': 1000000}}
          },
     'd4rl:maze2d-open-v0':
         {'observation_size': 4,
          'action_size': 2,
          'datasets': {
              '1k': {'name': 'd4rl:maze2d-open-v0',
                     'split': 1000},
              '10k': {'name': 'd4rl:maze2d-open-v0',
                      'split': 10000},
              '100k': {'name': 'd4rl:maze2d-open-v0',
                       'split': 100000},
              '1m': {'name': 'd4rl:maze2d-open-v0',
                     'split': 1000000}
          }
          }
     }
MUJOCO_ENV_CONFIGS = \
    {'Walker2d-v2':
         {'observation_size': 17,
          'action_size': 6,
          'datasets': {
              'random': {'name': 'd4rl:walker2d-random-v2', 'split': None},
              'expert': {'name': 'd4rl:walker2d-expert-v2', 'split': None},
              'medium': {'name': 'd4rl:walker2d-medium-v2', 'split': None},
              'medium-replay': {'name': 'd4rl:walker2d-medium-replay-v2',
                                'split': None},
              'medium-expert': {'name': 'd4rl:walker2d-medium-expert-v2',
                                'split': None}
          }
          },
     'Hopper-v2': {'observation_size': 11,
                   'action_size': 3,
                   'datasets': {'random': {'name': 'd4rl:hopper-random-v2',
                                           'split': None},
                                'expert': {'name': 'd4rl:hopper-expert-v2',
                                           'split': None},
                                'medium': {'name': 'd4rl:hopper-medium-v2',
                                           'split': None},
                                'medium-replay': {
                                    'name': 'd4rl:hopper-medium-replay-v2',
                                    'split': None},
                                'medium-expert': {
                                    'name': 'd4rl:hopper-medium-expert-v2',
                                    'split': None}
                                }
                   },
     'HalfCheetah-v2': {'observation_size': 17,
                        'action_size': 6,
                        'datasets': {
                            'random': {'name': 'd4rl:halfcheetah-random-v2',
                                       'split': None},
                            'expert': {'name': 'd4rl:halfcheetah-expert-v2',
                                       'split': None},
                            'medium': {'name': 'd4rl:halfcheetah-medium-v2',
                                       'split': None},
                            'medium-replay': {
                                'name': 'd4rl:halfcheetah-medium-replay-v2',
                                'split': None},
                            'medium-expert': {
                                'name': 'd4rl:halfcheetah-medium-expert-v2',
                                'split': None}
                        }
                        }
     }

ENV_PERFORMANCE_STATS = {
    'd4rl:maze2d-open-v0': {1: {'score_mean': 122.2, 'score_std': 10.61},
                            2: {'score_mean': 104.9, 'score_std': 22.19},
                            3: {'score_mean': 18.05, 'score_std': 14.85},
                            4: {'score_mean': 4.85, 'score_std': 8.62}},
    'd4rl:maze2d-medium-v1': {1: {'score_mean': 245.55, 'score_std': 272.75},
                              2: {'score_mean': 203.75, 'score_std': 252.61},
                              3: {'score_mean': 256.65, 'score_std': 260.16},
                              4: {'score_mean': 258.55, 'score_std': 262.81}},
    'd4rl:maze2d-umaze-v1': {1: {'score_mean': 235.5, 'score_std': 35.45},
                             2: {'score_mean': 197.75, 'score_std': 58.21},
                             3: {'score_mean': 23.4, 'score_std': 73.24},
                             4: {'score_mean': 3.2, 'score_std': 9.65}},
    'd4rl:maze2d-large-v1': {1: {'score_mean': 231.35, 'score_std': 268.37},
                             2: {'score_mean': 160.8, 'score_std': 201.97},
                             3: {'score_mean': 50.65, 'score_std': 76.94},
                             4: {'score_mean': 9.95, 'score_std': 9.95}},
    'd4rl:maze2d-open-dense-v0': {1: {'score_mean': 127.18, 'score_std': 9.17},
                                  2: {'score_mean': 117.53,
                                      'score_std': 10.21},
                                  3: {'score_mean': 63.96, 'score_std': 16.03},
                                  4: {'score_mean': 26.82, 'score_std': 9.19}},
    'd4rl:maze2d-medium-dense-v1': {
        1: {'score_mean': 209.25, 'score_std': 190.66},
        2: {'score_mean': 192.36, 'score_std': 193.29},
        3: {'score_mean': 225.54, 'score_std': 183.33},
        4: {'score_mean': 232.94, 'score_std': 184.62}},
    'd4rl:maze2d-umaze-dense-v1': {
        1: {'score_mean': 240.22, 'score_std': 25.1},
        2: {'score_mean': 201.12, 'score_std': 21.35},
        3: {'score_mean': 121.94, 'score_std': 10.71},
        4: {'score_mean': 45.5, 'score_std': 44.53}},
    'd4rl:maze2d-large-dense-v1': {
        1: {'score_mean': 168.83, 'score_std': 225.78},
        2: {'score_mean': 239.1, 'score_std': 208.43},
        3: {'score_mean': 204.39, 'score_std': 197.96},
        4: {'score_mean': 90.89, 'score_std': 70.61}},
    'HalfCheetah-v2': {1: {'score_mean': 1169.13, 'score_std': 80.45},
                       2: {'score_mean': 1044.39, 'score_std': 112.61},
                       3: {'score_mean': 785.88, 'score_std': 303.59},
                       4: {'score_mean': 94.79, 'score_std': 40.88}},
    'Hopper-v2': {1: {'score_mean': 1995.84, 'score_std': 794.71},
                  2: {'score_mean': 1466.71, 'score_std': 497.1},
                  3: {'score_mean': 1832.43, 'score_std': 560.86},
                  4: {'score_mean': 236.51, 'score_std': 1.09}},
    'Walker2d-v2': {1: {'score_mean': 2506.9, 'score_std': 689.45},
                    2: {'score_mean': 811.28, 'score_std': 321.66},
                    3: {'score_mean': 387.01, 'score_std': 42.82},
                    4: {'score_mean': 162.7, 'score_std': 102.14}}
}

ENV_CONFIGS = {**MAZE_ENV_CONFIGS, **MUJOCO_ENV_CONFIGS}
MIN_PRE_TRAINED_LEVEL = 1
MAX_PRE_TRAINED_LEVEL = 4
