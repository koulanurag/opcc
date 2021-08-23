import wandb
import cque.config

if __name__ == '__main__':

    # get data from API
    api = wandb.Api()
    # extract relevant data
    data = {}
    for runs in [api.runs(cque.config.MAZE_BASE_PROJECT_URL)]:
        for run in runs:
            env_name = run.config['env-name']
            data[env_name] = {'wandb_run_path': '/'.join(run.path)}

    for env_name, twins in [("halfcheetah-random-v2", ["halfcheetah-expert-v2", "halfcheetah-medium-v2",
                                                       "halfcheetah-medium-replay-v2", "halfcheetah-expert-v2"]),
                            ("walker2d-random-v2", ["walker2d-expert-v2", "walker2d-medium-v2",
                                                    "walker2d-medium-replay-v2", "walker2d-medium-expert-v2"]),
                            ("hopper-random-v2", ["hopper-expert-v2", "hopper-medium-v2",
                                                  "hopper-medium-replay-v2", "hopper-medium-expert-v2"])]:
        for twin in twins:
            data['d4rl:' + twin] = data['d4rl:' + env_name]

    print(data)
