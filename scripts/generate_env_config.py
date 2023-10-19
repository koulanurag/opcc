import os

import gym

import opcc.config

if __name__ == "__main__":
    maze2d_data, other_data, mujoco_data, adroit_data = {}, {}, {}, {}
    for env_name in os.listdir(opcc.config.ASSETS_DIR):
        env = gym.make(env_name)
        obs_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        env.close()
        if "maze" in env_name.lower():
            maze2d_data[env_name] = {
                "actor_kwargs": {
                    "state_dim": obs_size,
                    "action_dim": action_size,
                    "max_action": 1,
                    "hidden_dim": 64,
                },
                "datasets": {
                    "1k": {"name": env_name, "split": 1000},
                    "10k": {"name": env_name, "split": 10000},
                    "100k": {"name": env_name, "split": 100000},
                    "1m": {"name": env_name, "split": 1000000},
                },
            }
        elif env_name in ["HalfCheetah-v2", "Walker2d-v2", "Hopper-v2"]:
            mujoco_data[env_name] = {
                "actor_kwargs": {
                    "state_dim": obs_size,
                    "action_dim": action_size,
                    "max_action": 1,
                    "hidden_dim": 64,
                },
                "datasets": {
                    x: {
                        "name": "d4rl:{}-{}-v2".format(
                            env_name.lower().split("-")[0], x
                        ),
                        "split": None,
                    }
                    for x in [
                        "random",
                        "expert",
                        "medium",
                        "medium-replay",
                        "medium-expert",
                    ]
                },
            }
        elif env_name in ["d4rl:pen-v0", "d4rl:hammer-v0", "d4rl:door-v0"]:
            adroit_data[env_name] = {
                "actor_kwargs": {
                    "state_dim": obs_size,
                    "action_dim": action_size,
                    "max_action": 1,
                    "hidden_dim": 256,
                },
                "datasets": {
                    x: {
                        "name": "d4rl:{}-{}-v2".format(
                            env_name.lower().split("-")[0], x
                        ),
                        "split": None,
                    }
                    for x in [
                        "random",
                        "expert",
                        "medium",
                        "medium-replay",
                        "medium-expert",
                    ]
                },
            }
        else:
            pass

    print("MAZE2D DATA:\n{}\n*********".format(maze2d_data))
    print("MUJOCO DATA:\n{}\n*********".format(mujoco_data))
    print("ADROIT DATA:\n{}\n*********".format(adroit_data))
    print("OTHER DATA:\n{}\n*********".format(other_data))
