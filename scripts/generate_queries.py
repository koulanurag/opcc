"""
Usage: python generate_queries.py --env-name d4rl:maze2d-open-v0 --dataset-name 1k --policy-ids 1,2,3,4
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
import torch
from tqdm import tqdm
import opcc
import wandb
from kd import KDTree
from opcc.config import ASSETS_DIR


@torch.no_grad()
def mc_return(env, sim_state, init_action, horizon, policy, max_episodes):
    assert horizon <= env._max_episode_steps

    sim_state = np.array(sim_state).astype("float64")
    init_action = np.array(init_action).astype("float32")

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
                obs = torch.tensor(obs).unsqueeze(0).double()
                action = policy(obs).data.cpu().numpy()[0]
            obs, reward, done, info = env.step(action.astype("float32"))
            step_count += 1
            score += reward

        expected_score.append(score)
        expected_step.append(step_count)

    return expected_score, expected_step


@torch.no_grad()
def generate_query_states(env, policies, max_transaction_count, args):
    # create collection of states
    env_states = []

    with tqdm(total=max_transaction_count, desc="Generating Query States") as pbar:
        while len(env_states) < max_transaction_count:
            policy = random.choice(list(policies.values()))
            obs = env.reset()

            done = False
            while not done:
                save = random.random() >= args.save_prob
                if save:
                    env_states.append(
                        (obs.tolist(), env.sim.get_state().flatten().tolist())
                    )
                    pbar.update(1)

                action = policy(torch.tensor(obs).unsqueeze(0).double())
                noise = torch.normal(0, args.noise, size=action.shape).double()
                step_action = (action + noise).data.cpu().numpy()[0]
                step_action = step_action.astype("float32")
                obs, _, done, info = env.step(step_action)
    return env_states


@torch.no_grad()
def evaluate_queries(env, candidate_states, policies, args):
    _overall_data = defaultdict(lambda: [])
    _queries = {}
    total_query_count = 0
    policy_ids = sorted(policies.keys())
    for i, policy_id_a in enumerate(tqdm(policy_ids, desc="Policy-A")):
        policy_a = policies[policy_id_a]
        for policy_id_b in tqdm(args.policy_ids[i + 1 :], desc="Policy-B"):
            policy_b = policies[policy_id_b]

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

            with tqdm(
                total=args.per_policy_comb_query, desc="Generating Query"
            ) as pbar:
                while (
                    query_count < args.per_policy_comb_query
                    and ignore_count < args.ignore_stuck_count
                ):
                    same_state = random.choices([True, False], weights=[0.2, 0.8], k=1)[
                        0
                    ]

                    # query-a attributes
                    (obs_a, sim_state_a) = random.choice(candidate_states)
                    action_a = env.action_space.sample()

                    # query-b attributes
                    if same_state:
                        obs_b = deepcopy(obs_a)
                        sim_state_b = deepcopy(sim_state_a)
                    else:
                        obs_b, sim_state_b = random.choice(candidate_states)
                    action_b = env.action_space.sample()

                    # evaluate
                    horizon = random.choice(args.horizons)
                    return_a, horizon_a = mc_return(
                        env, sim_state_a, action_a, horizon, policy_a, args.eval_runs
                    )
                    return_b, horizon_b = mc_return(
                        env, sim_state_b, action_b, horizon, policy_b, args.eval_runs
                    )
                    return_a_mean = np.mean(return_a)
                    return_b_mean = np.mean(return_b)
                    horizon_a_mean = np.mean(horizon_a)
                    horizon_b_mean = np.mean(horizon_b)

                    # ignore ambiguous queries
                    if (abs(return_a_mean - return_b_mean) <= args.ignore_delta) or (
                        min(return_b) <= max(return_a) <= max(return_b)
                        or min(return_b) <= min(return_a) <= max(return_b)
                    ):
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

                        # update progress bar
                        pbar.update(1)
                        pbar.set_description(
                            f"Ignore Count:{ignore_count} "
                            f"| Total Query Count: {total_query_count}"
                            f"| Generating Query"
                        )

                        # log for tracking progress
                        if args.use_wandb:
                            wandb.log({"query-count": total_query_count})

            _key = ((args.env_name, policy_id_a), (args.env_name, policy_id_b))
            _queries[_key] = {
                "obs_a": np.array(obss_a),
                "action_a": np.array(actions_a),
                "obs_b": np.array(obss_b),
                "action_b": np.array(actions_b),
                "horizon": np.array(horizons),
                "target": np.array(targets),
                "info": {
                    "return_a": np.array(returns_a),
                    "return_b": np.array(returns_b),
                    "return_list_a": np.array(returns_list_a),
                    "return_list_b": np.array(returns_list_b),
                    "state_a": np.array(sim_states_a),
                    "state_b": np.array(sim_states_b),
                    "runs": args.eval_runs,
                    "horizon_a": horizons_a,
                    "horizon_b": horizons_b,
                },
            }

            # save data separately for ease of visualization
            _overall_data["obs-a"] += obss_a
            _overall_data["obs-b"] += obss_b
            _overall_data["action-a"] += actions_a
            _overall_data["action-b"] += actions_b
            _overall_data["return-a"] += returns_a
            _overall_data["return-b"] += returns_b
            _overall_data["target"] += targets
            _overall_data["horizon"] += horizons
            _overall_data["horizon-a"] += horizons_a
            _overall_data["horizon-b"] += horizons_b
    return _queries, _overall_data


def main():
    # Lets gather arguments
    parser = argparse.ArgumentParser(description="Generate queries")
    parser.add_argument("--env-name", default="d4rl:maze2d-open-v0")
    parser.add_argument("--eval-runs", type=int, default=2)
    parser.add_argument(
        "--noise",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--ignore-delta-per-horizons",
        default=20,
        nargs="+",
        required=True,
        type=int,
        help="ignore query if difference between two sides"
        " of query is less than it.",
    )
    parser.add_argument(
        "--horizons", nargs="+", help="horizon lists", type=int, required=True
    )
    parser.add_argument(
        "--policy-ids", nargs="+", help="policy id lists", type=int, required=True
    )
    parser.add_argument("--use-wandb", action="store_true", default=False)
    parser.add_argument(
        "--max-trans-count",
        type=int,
        default=1000,
        help="maximum number of transition count",
    )
    parser.add_argument(
        "--ignore-stuck-count",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--save-prob",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--per-policy-comb-query",
        type=int,
        default=100,
    )

    # Process arguments
    args = parser.parse_args()
    assert len(args.ignore_delta_per_horizon)== len(args.horizons)

    if args.use_wandb:
        wandb.init(project="opcc", config={"env_name": args.env_name}, save_code=True)

    # summarize arguments
    print(
        f'\n\n{"=" * 100}\n'
        + f'{"Argument Name":<50}\tValue'
        + f'\n{"------------":<50}\t-----\n'
        + "\n".join(
            f"{arg_name:<50}\t{getattr(args, arg_name)}" for arg_name in vars(args)
        )
        + f'\n\n{"=" * 100}\n'
    )

    # seed
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # generate queries
    env = gym.make(args.env_name)
    env.action_space.seed(0)
    env.seed(0)

    policies = {
        policy_id: opcc.get_policy(args.env_name, policy_id)[0]
        for policy_id in args.policy_ids
    }

    candidate_states = generate_query_states(env, policies, args.max_trans_count, args)
    queries, overall_data = evaluate_queries(env, candidate_states, policies, args)

    # _path = os.path.join(ASSETS_DIR, args.env_name, "queries.p")
    # queries = pickle.load(open(_path, "rb"))
    # overall_data = defaultdict(lambda: [])
    # for value_dict in queries.values():
    #     for _key, value in value_dict.items():
    #         try:
    #             overall_data[_key.replace("_","-")] += value.tolist()
    #         except:
    #             overall_data[_key.replace("_","-")] += value

    # save queries
    _path = os.path.join(ASSETS_DIR, args.env_name, "queries.p")
    os.makedirs(os.path.join(ASSETS_DIR, args.env_name), exist_ok=True)
    pickle.dump(queries, open(_path, "wb"))
    if args.use_wandb:
        wandb.save(_path)

    # estimate distance of queries from datasets
    query_obs_action_a = np.concatenate((overall_data["obs-a"], overall_data["action-a"]), 1)
    query_obs_action_b = np.concatenate(
        (overall_data["obs-b"], overall_data["action-b"]), 1
    )
    for dataset_name in tqdm(
        opcc.get_dataset_names(args.env_name),
        desc=" Query distance from dataset| Datasets",
    ):
        dataset = opcc.get_qlearning_dataset(args.env_name, dataset_name)
        kd_tree = KDTree(np.concatenate((dataset["observations"], dataset["actions"]), 1))
        kd_data_a = kd_tree.get_knn_batch(query_obs_action_a, k=1)
        kd_data_b = kd_tree.get_knn_batch(query_obs_action_b, k=1)

        distance_a = [list(_.values())[0] for _ in kd_data_a]
        distance_b = [list(_.values())[0] for _ in kd_data_b]

        # store distances
        overall_data["query_a-{}".format(dataset_name)] = distance_a
        overall_data["query_b-{}".format(dataset_name)] = distance_b

    # visualize
    if args.use_wandb:
        overall_df = pd.DataFrame(data=overall_data)
        return_fig = px.scatter(
            overall_df,
            x="return-a",
            y="return-b",
            color="target",
            marginal_x="histogram",
            marginal_y="histogram",
            symbol="horizon",
        )

        distances_data = []
        dataset_names_data = []
        for dataset_name in opcc.get_dataset_names(args.env_name):
            distances_data += overall_data["query_a-{}".format(dataset_name)]
            distances_data += overall_data["query_b-{}".format(dataset_name)]
            dataset_names_data += [
                dataset_name for _ in range(len(overall_data["obs-a"]))
            ]
            dataset_names_data += [
                dataset_name for _ in range(len(overall_data["obs-a"]))
            ]

        distance_df = pd.DataFrame(
            data={"distance": distances_data, "dataset": dataset_names_data}
        )
        distance_fig = px.histogram(distance_df, x="distance", color="dataset")
        wandb.log(
            {
                "query-values-scatter": return_fig,
                "distance-histogram": distance_fig,
                "query-data": wandb.Table(dataframe=overall_df),
            }
        )

    # close env
    env.close()


if __name__ == "__main__":
    main()
