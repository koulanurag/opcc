# -*- coding: utf-8 -*-
import argparse
import os
import pickle
from pathlib import Path

import gym
import numpy as np
import torch
from tqdm import tqdm

import opcc
from opcc.config import ENV_CONFIGS
from opcc.config import MAX_PRE_TRAINED_LEVEL, MIN_PRE_TRAINED_LEVEL


def generate_env_stats(env_name, test_episodes, stats_dir, no_cache=False,
                       render=False):
    env_stats_path = os.path.join(stats_dir, env_name + '.p')
    if not no_cache:
        if os.path.exists(env_stats_path):
            print('Using Existing stats from :{}'.format(env_stats_path))
            return pickle.load(open(env_stats_path, 'rb'))

    env_info = {}
    for pre_trained_id in tqdm(range(MIN_PRE_TRAINED_LEVEL,
                                     MAX_PRE_TRAINED_LEVEL + 1)):
        model, _ = opcc.get_policy(env_name, pre_trained_id)
        episode_rewards = []
        for episode_i in range(test_episodes):
            env = gym.make(env_name)
            env.seed(episode_i)
            done = False
            episode_reward = 0
            obs = env.reset()
            while not done:
                if render:
                    env.render()
                action = model.actor(torch.tensor(obs).unsqueeze(0).float())
                action = action.data.numpy()[0]
                obs, reward, done, step_info = env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)
            env.close()

        episode_rewards = np.array(episode_rewards)
        mean = round(episode_rewards.mean(), 2)
        std = round(episode_rewards.std(), 2)
        env_info[pre_trained_id] = {'score_mean': mean, 'score_std': std}

    pickle.dump(env_info, open(env_stats_path, 'wb'))
    return env_info


def markdown_pre_trained_scores(env_info):
    # create markdown for the table:
    msg = "| Environment Name |"
    for i in range(MIN_PRE_TRAINED_LEVEL, MAX_PRE_TRAINED_LEVEL + 1):
        pre_info = '(best)' if i == MIN_PRE_TRAINED_LEVEL else (
            '(worst)' if i == MAX_PRE_TRAINED_LEVEL else '')
        msg += "`pre_trained={}` {} ".format(i, pre_info) + "|"

    msg += '\n'
    msg += "|" + " | ".join(
        ":------:" for _ in range(MAX_PRE_TRAINED_LEVEL + 1)) + ' | ' + '\n'

    for env_name in tqdm(env_info):
        msg += "|{}|".format("`{}`".format(env_name))
        for i in range(MIN_PRE_TRAINED_LEVEL, MAX_PRE_TRAINED_LEVEL + 1):
            if i in env_info[env_name]:
                msg += '{}±{} |'.format(env_info[env_name][i]['score_mean'],
                                        env_info[env_name][i]['score_std'])
            else:
                msg += '- |'
        msg += '\n'
    return msg


if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='Generate stats'
                                                 ' for environment')
    parser.add_argument('--env-name', required=False, type=str,
                        help='Name of the environment',
                        default='d4rl:maze2d-open-v0')
    parser.add_argument('--test-episodes', required=False, default=20,
                        type=int, help='No. of episodes for evaluation')
    parser.add_argument('--all-envs', default=False, action='store_true',
                        help="Generate stats for all envs "
                             "(default: %(default)s)")
    parser.add_argument('--no-cache', default=False, action='store_true',
                        help="Doesn't use pre-generated stats "
                             " (default: %(default)s)")
    parser.add_argument('--stats-dir', type=str,
                        default=os.path.join(str(Path.home()), '.opcc',
                                             'generated_stats'))
    parser.add_argument('--render', default=False, action='store_true',
                        help="Renders the environment while evaluating "
                             " (default: %(default)s)")
    args = parser.parse_args()
    os.makedirs(args.stats_dir, exist_ok=True)
    stats_info = {}

    for env_name in tqdm(ENV_CONFIGS.keys()
                         if args.all_envs else [args.env_name]):
        stats_info[env_name] = generate_env_stats(env_name, args.test_episodes,
                                                  args.stats_dir,
                                                  no_cache=args.no_cache,
                                                  render=args.render)
    table_markdown = markdown_pre_trained_scores(stats_info)
    print(table_markdown)
