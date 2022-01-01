"""
Usage: python generate_queries.py --env-name d4rl:maze2d-open-v0
                                  --dataset-name 1k --policy-ids 1,2,3,4
"""

import argparse
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy

import gym
import numpy as np
import pandas as pd
import plotly.express as px
import policybazaar
import torch
import wandb

from cque.config import CQUE_DIR


def mc_return(env, sim_state, init_action, horizon, policy, max_episodes):
    assert horizon <= env._max_episode_steps

    expected_score = []
    expected_step = []
    for ep_i in range(max_episodes):
        env.reset()
        env.sim.set_state_from_flattened(sim_state)
        env.sim.forward()

        obs, reward, done, info = env.step(init_action)
        score = reward
        step_count = 1

        while not done and step_count < horizon:
            with torch.no_grad():
                obs = torch.tensor(obs).unsqueeze(0).float()
                action = policy.actor(obs).data.cpu().numpy()[0]
            obs, reward, done, info = env.step(action)
            step_count += 1
            score += reward

        expected_score.append(score)
        expected_step.append(step_count)

    return expected_score, expected_step


if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Generate queries')
    parser.add_argument('--env-name', default='d4rl:maze2d-open-v0')
    parser.add_argument('--eval-runs', type=int, default=2)
    parser.add_argument('--noise', type=float, default=0.05, )
    parser.add_argument('--ignore-delta', type=float, default=20,
                        help='ignore query if difference between two sides '
                             'of query is less than it.')
    parser.add_argument('--horizons', nargs='+', help='horizon lists',
                        type=int, required=True)
    parser.add_argument('--policy-ids', nargs='+', help='policy id lists',
                        type=int, required=True)
    parser.add_argument('--use-wandb', action='store_true', default=False)
    parser.add_argument('--max-transaction-count', type=int, default=1000, )
    parser.add_argument('--ignore-stuck-count', type=int, default=200, )
    parser.add_argument('--save-prob', type=float, default=0.7, )
    parser.add_argument('--per-policy-comb-query', type=int, default=100, )

    # Process arguments
    args = parser.parse_args()
    if args.use_wandb:
        wandb.init(project='cque', config={'env-name': args.env_name})

    # seed
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # create collection of states
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
                env_states.append((obs, env.sim.get_state().flatten().tolist()))
            action = model.actor(torch.tensor(obs).unsqueeze(0).float())
            noise = torch.normal(0, args.noise, size=action.shape)
            step_action = (action + noise).data.cpu().numpy()[0]
            obs, _, done, info = env.step(step_action)

    # evaluate queries
    overall_data = defaultdict(lambda: [])
    queries = {}
    total_query_count = 0
    for i, policy_id_a in enumerate(args.policy_ids):
        policy_a, _ = policybazaar.get_policy(args.env_name, policy_id_a)
        for policy_id_b in args.policy_ids[i + 1:]:
            policy_b, _ = policybazaar.get_policy(args.env_name, policy_id_b)

            # core attributes
            obss_a = []
            obss_b = []
            actions_a = []
            actions_b = []
            horizons = []
            targets = []

            # info attributes
            returns_a = []
            returns_b = []
            returns_list_a = []
            returns_list_b = []
            horizons_a = []
            horizons_b = []
            sim_states_a = []
            sim_states_b = []

            query_count = 0
            ignore_count = 0
            while query_count < args.per_policy_comb_query and ignore_count < args.ignore_stuck_count:
                same_state = random.choices([True, False], weights=[0.2, 0.8], k=1)[0]

                # query-a attributes
                (obs_a, sim_state_a) = random.choice(env_states)
                action_a = env.action_space.sample()

                # query-b attributes
                if same_state:
                    obs_b = deepcopy(obs_a)
                    sim_state_b = deepcopy(sim_state_a)
                else:
                    obs_b, sim_state_b = random.choice(env_states)
                action_b = env.action_space.sample()

                # evaluate
                horizon = random.choice(args.horizons)
                return_a, horizon_a = mc_return(env, sim_state_a, action_a, horizon, policy_a, args.eval_runs)
                return_b, horizon_b = mc_return(env, sim_state_b, action_b, horizon, policy_b, args.eval_runs)
                return_a_mean, return_b_mean = np.mean(return_a), np.mean(return_b)
                horizon_a_mean, horizon_b_mean = np.mean(horizon_a), np.mean(horizon_b)
                if (abs(return_a_mean - return_b_mean) <= args.ignore_delta) \
                        and ((max(return_a) < min(return_b)) or (min(return_a) > max(return_b))):
                    ignore_count += 1
                    continue
                else:
                    obss_a.append(obs_a)
                    obss_b.append(obs_b)
                    actions_a.append(action_a)
                    actions_b.append(action_b)
                    horizons.append(horizon)
                    targets.append(return_a_mean < return_b_mean)

                    returns_a.append(return_a_mean)
                    returns_b.append(return_b_mean)
                    returns_list_a.append(return_a)
                    returns_list_b.append(return_b)
                    sim_states_a.append(sim_state_a)
                    sim_states_b.append(sim_state_b)
                    horizons_a.append(horizon_a_mean)
                    horizons_b.append(horizon_b_mean)

                    query_count += 1
                    total_query_count += 1

                    # log for tracking progress
                    if args.use_wandb:
                        wandb.log({'query-count': total_query_count})

            _key = ((args.env_name, policy_id_a), (args.env_name, policy_id_b))
            queries[_key] = {'obs_a': np.array(obss_a),
                             'action_a': np.array(actions_a),
                             'obs_b': np.array(obss_b),
                             'action_b': np.array(actions_b),
                             'horizon': np.array(horizons),
                             'target': np.array(targets),
                             'info': {'return_a': np.array(returns_a),
                                      'return_b': np.array(returns_b),
                                      'return_list_a': np.array(returns_list_a),
                                      'return_list_b': np.array(returns_list_b),
                                      'state_a': np.array(sim_states_a),
                                      'state_b': np.array(sim_states_b),
                                      'runs': args.eval_runs,
                                      'horizon_a': horizons_a,
                                      'horizon_b': horizons_b}}

            # save data separately for ease of visualization
            overall_data['obs-a'] += obss_a
            overall_data['obs-b'] += obss_b
            overall_data['return-a'] += returns_a
            overall_data['return-b'] += returns_b
            overall_data['target'] += targets
            overall_data['horizon'] += horizons
            overall_data['horizon-a'] += horizons_a
            overall_data['horizon-b'] += horizons_b

    # visualize
    if args.use_wandb:
        df = pd.DataFrame(data=overall_data)
        fig = px.scatter(df, x='return-a', y='return-b', color='target',
                         marginal_x="histogram", marginal_y="histogram",
                         symbol='horizon')
        wandb.log({'query-values-scatter': fig,
                   'query-data': df})

    # save queries
    _path = os.path.join(CQUE_DIR, args.env_name, 'queries.p')
    os.makedirs(os.path.join(CQUE_DIR, args.env_name), exist_ok=True)
    pickle.dump(queries, open(_path, 'wb'))
    if args.use_wandb:
        wandb.save(_path)

    # close env
    env.close()
