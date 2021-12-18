import argparse
import os
import pickle
import random
from copy import deepcopy

import gym
import numpy as np
import pandas as pd
import plotly.express as px
import policybazaar
import torch
import wandb
from cque.config import CQUE_DIR


def mc_return(env, init_action, horizon, policy, max_episodes):
    expected_score = 0
    for ep_i in range(max_episodes):

        _env = deepcopy(env)
        obs, reward, done, info = _env.step(init_action)
        if 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
            done = False

        score = reward
        step_count = 1

        while not done and step_count < horizon:
            action = policy.actor(torch.tensor(obs).unsqueeze(0).float())
            obs, reward, done, info = _env.step(action.data.cpu().numpy()[0])

            if 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
                done = False
            step_count += 1
            score += reward

        expected_score += score

    return expected_score / max_episodes


if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Generate queries')
    parser.add_argument('--env-name', required=False, default='d4rl:maze2d-open-v0')
    parser.add_argument('--max-eval-episodes', type=int, default=2, required=False)
    parser.add_argument('--noise', type=float, default=0.05, required=False)
    parser.add_argument('--ignore-delta', type=float, default=20, required=False)
    parser.add_argument('--horizons', nargs='+', help='<Required> Set flag', type=int, required=True)
    parser.add_argument('--policy_ids', nargs='+', help='<Required> Set flag', type=int, required=True)
    parser.add_argument('--use-wandb', action='store_true', default=False)
    parser.add_argument('--max-transaction-count', type=int, default=1000, required=False)
    parser.add_argument('--ignore-stuck-count', type=int, default=200, required=False)
    parser.add_argument('--save-prob', type=float, default=0.7, required=False)
    parser.add_argument('--per-policy-comb-query', type=int, default=100, required=False)

    # Process arguments
    args = parser.parse_args()
    if args.use_wandb:
        wandb.init(project='cque', config={'env-name': args.env_name})

    np.random.seed(0)
    random.seed(0)

    env_states = []
    env = gym.make(args.env_name)
    while len(env_states) < args.max_transaction_count:
        policy_id = random.choice(args.policy_ids)
        model, model_info = policybazaar.get_policy(args.env_name, policy_id)
        obs = env.reset()

        done = False
        while not done:
            save = random.random() >= args.save_prob
            if save:
                env_states.append((obs, deepcopy(env)))
            action = model.actor(torch.tensor(obs).unsqueeze(0).float())
            noise = torch.normal(0, args.noise, size=action.shape)
            # noise = 0
            step_action = (action + noise).data.cpu().numpy()[0]
            obs, _, done, info = env.step(step_action)

    # evaluate queries
    overall_data = {'q-value-a': [], 'q-value-b': [], 'target': [], 'horizon-a': [], 'horizon-b': [],
                    'horizons': []}
    queries = {}
    for i, id_a in enumerate(args.policy_ids):
        policy_a, _ = policybazaar.get_policy(args.env_name, id_a)
        for id_b in args.policy_ids[i:]:
            policy_b, _ = policybazaar.get_policy(args.env_name, id_b)

            states_a = []
            states_b = []
            actions_a = []
            actions_b = []
            horizons = []
            horizons_a = []
            horizons_b = []
            targets_a = []
            targets_b = []
            targets = []

            query_count = 0
            ignore_count = 0
            while query_count < args.per_policy_comb_query and ignore_count < args.ignore_stuck_count:
                same_state = random.choice([True, False])
                same_action = False

                (state_a, state_a_env), action_a = random.choice(env_states), env.action_space.sample()
                (_state_b, _state_b_env), _action_b = random.choice(env_states), env.action_space.sample()
                state_a_env = deepcopy(state_a_env)

                if same_state:
                    state_b = state_a
                    state_b_env = deepcopy(state_a_env)
                else:
                    state_b = _state_b
                    state_b_env = deepcopy(_state_b_env)

                if same_action:
                    action_b = action_a
                else:
                    action_b = _action_b

                same_horizon = True
                if same_horizon:
                    _horizon = random.choice(args.horizons)
                    horizon_a, horizon_b = _horizon, _horizon
                else:
                    horizon_a, horizon_b = np.random.choice(args.horizons, replace=False, size=(2,))

                # evaluate
                target_a = mc_return(state_a_env, action_a, horizon_a, policy_a, args.max_eval_episodes)
                target_b = mc_return(state_b_env, action_b, horizon_b, policy_b, args.max_eval_episodes)
                print(target_a, target_b)
                if abs(target_a - target_b) <= args.ignore_delta:
                    ignore_count += 1
                    continue
                else:
                    states_a.append(state_a)
                    states_b.append(state_b)
                    actions_a.append(action_a)
                    actions_b.append(action_b)
                    horizons.append((horizon_a, horizon_b))
                    horizons_a.append(horizon_a)
                    horizons_b.append(horizon_b)
                    targets_a.append(target_a)
                    targets_b.append(target_b)
                    targets.append(target_a < target_b)

                    query_count += 1

            _key = ((args.env_name, id_a), (args.env_name, id_b))
            queries[_key] = (np.array(states_a), np.array(actions_a), np.array(horizons_a),
                             np.array(states_b), np.array(actions_b), np.array(horizons_b),
                             np.array(targets_a), np.array(targets_b), np.array(targets))

            # save data separately for ease of visualization
            overall_data['q-value-a'] += targets_a
            overall_data['q-value-b'] += targets_b
            overall_data['target'] += targets
            overall_data['horizon-a'] += horizons_a
            overall_data['horizons'] += horizons
            overall_data['horizon-b'] += horizons_b

    # visualize
    if args.use_wandb:
        df = pd.DataFrame(data=overall_data)
        fig = px.scatter(df, x='q-value-a', y='q-value-b', color='target',
                         marginal_x="histogram", marginal_y="histogram",
                         symbol='horizons')
        wandb.log({'query-values-scatter': fig,
                   'query-data': df,
                   'query-count': len(overall_data['q-value-a'])})

    # save queries
    _path = os.path.join(CQUE_DIR, args.env_name, 'queries.p')
    os.makedirs(os.path.join(CQUE_DIR, args.env_name), exist_ok=True)
    pickle.dump(queries, open(_path, 'wb'))
    if args.use_wandb:
        wandb.save(_path)

    # close env
    env.close()
