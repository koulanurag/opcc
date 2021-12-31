import wandb

import cque.config

if __name__ == '__main__':

    # get data from API
    api = wandb.Api()
    # extract relevant data
    maze2d_data, other_data, mujoco_data = {}, {}, {}
    for runs in [api.runs(cque.config.MAZE_BASE_PROJECT_URL)]:
        for run in runs:
            env_name = run.config['env-name']
            if 'maze' in env_name.lower():
                maze2d_data[env_name] = {'wandb_run_path': '/'.join(run.path),
                                         'datasets': {'1k': {'name': env_name,
                                                             'split': 1000},
                                                      '10k': {'name': env_name,
                                                              'split': 10000},
                                                      '100k': {'name': env_name,
                                                               'split': 100000}}}
            elif env_name in ['HalfCheetah-v2', 'Walker2d-v2', 'Hopper-v2']:
                mujoco_data[env_name] = {'wandb_run_path': '/'.join(run.path),
                                         'datasets':
                                             {x: {'name': 'd4rl:{}-{}-v2'.format(env_name.lower().split('-')[0], x),
                                                  'split': None}
                                              for x in ['random', 'expert', 'medium', 'medium-replay',
                                                        'medium-expert']}}
            else:
                other_data[env_name] = {'wandb_run_path': '/'.join(run.path)}

    print('MAZE2D DATA:\n{}\n*********'.format(maze2d_data))
    print('MUJOCO DATA:\n{}\n*********'.format(mujoco_data))
    print('OTHER DATA:\n{}\n*********'.format(other_data))
