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
    print(data)
