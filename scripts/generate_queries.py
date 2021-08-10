import argparse
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
    parser.add_argument('--gamma', type=float, default=0.99, required=False)

    # Process arguments
    args = parser.parse_args()
    wandb.init(project='cque', config={'env-name': args.env_name})

    queries = {}
    env = gym.make(args.env_name)

    for id_a in range(1, 5):
        for id_b in range(1, 5):
            if id_a == id_b:
                break

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

            for seed in range(2):
                env.seed(seed)
                obs = env.reset().tolist()
                root_action_a = env.action_space.sample().tolist()
                root_action_b = env.action_space.sample().tolist()
                policy_a, _ = policybazaar.get_policy(args.env_name, id_a)
                policy_b, _ = policybazaar.get_policy(args.env_name, id_b)

                info = {}

                for horizon in [10, 50, 100]:
                    # evaluate policy
                    for policy, root_action, name in [(policy_a, root_action_a, 'policy_a'),
                                                      (policy_b, root_action_b, 'policy_b')]:
                        episode_rewards = 0
                        for _ in range(args.max_episodes):
                            episode_reward = 0
                            env.seed(seed)
                            env.reset()
                            step_obs, step_reward, done, _ = env.step(root_action)
                            episode_reward += step_reward
                            for step_count in range(horizon):
                                action = policy.actor(torch.tensor(step_obs).unsqueeze(0).float())
                                action = action.data.cpu().numpy()[0]
                                step_obs, step_reward, done, _ = env.step(action)
                                episode_reward += args.gamma * step_reward
                            episode_rewards += episode_reward

                        target = episode_rewards / args.max_episodes
                        info[name] = target

                    # store data
                    observations_a.append(obs)
                    observations_b.append(obs)

                    actions_a.append(root_action_a)
                    actions_b.append(root_action_b)

                    gamma_a.append(args.gamma)
                    gamma_b.append(args.gamma)

                    horizon_a.append(horizon)
                    horizon_b.append(horizon)

                    targets_a.append(info['policy_a'])
                    targets_b.append(info['policy_b'])
                    targets.append(targets_a[-1] < targets_b[-1])

            _key = ((args.env_name, id_a), (args.env_name, id_b))
            queries[_key] = (observations_a, actions_a, horizon_a, gamma_a,
                             observations_b, actions_b, horizon_b, gamma_b,
                             targets_a, targets_b, targets)

    _path = os.path.join(CQUE_DIR, args.env_name, 'queries.p')
    os.makedirs(os.path.join(CQUE_DIR, args.env_name), exist_ok=True)
    pickle.dump(queries, open(_path, 'wb'))
    wandb.save(_path)

    # close env
    env.close()
