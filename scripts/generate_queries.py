import argparse

import numpy as np
import wandb
import policybazaar
import pickle
from cque.config import CQUE_DIR
import os
import gym
import torch

if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Generate queries')
    parser.add_argument('--env-name', required=False, default='d4rl:maze2d-open-v0')
    parser.add_argument('--max-episodes', type=int, default=2, required=False)
    parser.add_argument('--gamma', type=float, default=1, required=False)
    parser.add_argument('--episode-count', type=float, default=5, required=False)
    parser.add_argument('--horizons', '--list', nargs='+', help='<Required> Set flag', type=int, required=True)

    # Process arguments
    args = parser.parse_args()
    wandb.init(project='cque', config={'env-name': args.env_name})

    queries = {}
    overall_target_a = []
    overall_target_b = []
    overall_target = []
    overall_horizon = []
    env = gym.make(args.env_name)

    for id_a in range(1, 4):
        for id_b in range(1, 4):
            if id_a == id_b:
                continue

            observations_a = []
            observations_b = []
            actions_a = []
            actions_b = []
            gamma_a = []
            gamma_b = []
            horizon_a = []
            horizon_b = []
            targets_a = []
            targets_b = []
            targets = []

            for seed in range(args.episode_count):
                env.seed(seed)
                obs = env.reset().tolist()
                root_action_a = env.action_space.sample().tolist()
                root_action_b = env.action_space.sample().tolist()
                policy_a, _ = policybazaar.get_policy(args.env_name, id_a)
                policy_b, _ = policybazaar.get_policy(args.env_name, id_b)

                info = {}

                # evaluate policy
                horizons = args.horizons
                for policy, root_action, name in [(policy_a, root_action_a, 'policy_a'),
                                                  (policy_b, root_action_b, 'policy_b')]:
                    max_horizon = max(horizons)
                    episode_rewards = np.zeros((2, max_horizon))
                    for ep_i in range(args.max_episodes):
                        env.seed(seed)
                        step_obs = env.reset()
                        assert (obs == step_obs).all()
                        step_obs, step_reward, done, _ = env.step(np.array(root_action))
                        episode_rewards[ep_i, 0] = step_reward
                        for step_count in range(1, max_horizon):
                            action = policy.actor(torch.tensor(step_obs).unsqueeze(0).float())
                            action = action.data.cpu().numpy()[0]
                            step_obs, step_reward, done, _ = env.step(action)
                            episode_rewards[ep_i, step_count] = step_reward
                    info[name] = episode_rewards

                for horizon in horizons:
                    # store data
                    observations_a.append(obs)
                    observations_b.append(obs)

                    actions_a.append(root_action_a)
                    actions_b.append(root_action_b)

                    gamma_a.append(args.gamma)
                    gamma_b.append(args.gamma)

                    horizon_a.append(horizon)
                    horizon_b.append(horizon)

                    targets_a.append(info['policy_a'][:, :horizon].sum(1).mean())
                    targets_b.append(info['policy_b'][:, :horizon].sum(1).mean())
                    targets.append(targets_a[-1] < targets_b[-1])

            _key = ((args.env_name, id_a), (args.env_name, id_b))
            queries[_key] = (observations_a, actions_a, horizon_a, gamma_a,
                             observations_b, actions_b, horizon_b, gamma_b,
                             targets_a, targets_b, targets)
            overall_target_a += targets_a
            overall_target_b += targets_b
            overall_target += targets
            overall_horizon += horizon_a

    import plotly.express as px
    import pandas as pd

    df = pd.DataFrame(
        data={'q-value-a': overall_target_a, 'q-value-b': overall_target_b, 'target': overall_target,
              'horizon': overall_horizon})
    fig1 = px.scatter_3d(df, x='q-value-a', y='q-value-b', z='target', color='target', symbol='horizon')
    fig2 = px.scatter(df, x='q-value-a', y='q-value-b', color='target')
    wandb.log({'query-values-3d': fig1, 'query-values-scatter': fig2})
    _path = os.path.join(CQUE_DIR, args.env_name, 'queries.p')
    os.makedirs(os.path.join(CQUE_DIR, args.env_name), exist_ok=True)
    pickle.dump(queries, open(_path, 'wb'))
    wandb.save(_path)

    # close env
    env.close()
