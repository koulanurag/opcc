import os
import hashlib

GENERIC_TARGET = """
description: OPCC-TD3
target:
  service: sing
  name: msrresrchvc
"""

STATIC_YAML_SEGMENT = """
environment:
  image: amlt-sing/pytorch-1.11.0
  image_setup:

    # Setup Mujoco
    - sudo apt-get update
    - sudo apt install -y  -q libosmesa6-dev  libglew-dev  libgl1-mesa-glx libgl1-mesa-dev libgl1-mesa-glx libglfw3
    - echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
    - sudo apt install -y -q xpra
    - sudo apt install -y -q xserver-xorg-dev
    - sudo apt install -y -q patchelf
    - sudo apt install -y -q gcc-multilib
    - mkdir -p /home/aiscuser/.mujoco
    - wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
    - tar -xf mujoco210-linux-x86_64.tar.gz -C /home/aiscuser/.mujoco
    - rm -rf *.tar.gz*
    - echo 'export LD_LIBRARY_PATH=$$LD_LIBRARY_PATH:/home/aiscuser/.mujoco/mujoco210/bin' >> /home/aiscuser/.bashrc

    # pip dependencies
    - pip install protobuf==3.20.1
    - pip3 install wandb
    - pip3 install pyglet
    - pip3 install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
    - pip3 install "cython<3" --upgrade
    - echo 'export CPATH="/usr/include:$$CPATH"' >> /home/aiscuser/.bashrc
    - chmod -R 777 /opt/conda/lib/python3.8/site-packages/mujoco_py

storage:
  opcctd3:
    storage_account_name: anuragkoul
    container_name: opcctd3
    is_output: true
  d4rl:
    storage_account_name: anuragkoul
    container_name: d4rl
    is_output: true


code:
  local_dir: $CONFIG_DIR/../..
"""

MAZE2D_ENVS = [
    # 'maze2d-open-v0',
    # 'maze2d-umaze-v1',
    # 'maze2d-medium-v1',
    # 'maze2d-large-v1'
]
MAZE2D_DENSE_ENVS = [
    # 'maze2d-open-dense-v0',
    # 'maze2d-umaze-dense-v1',
    # 'maze2d-medium-dense-v1',
    # 'maze2d-large-dense-v1'
]

ANT_MAZE_ENVS = [
    "antmaze-umaze-v0",
    # 'antmaze-umaze-diverse-v0',
    "antmaze-medium-diverse-v0",
    # 'antmaze-medium-play-v0',
    "antmaze-large-diverse-v0",
    # 'antmaze-large-play-v0',
]

GYM_MUJOCO_ENVS = [
    # 'halfcheetah-random-v2',
    # 'halfcheetah-medium-v2',
    # 'halfcheetah-expert-v2',
    # 'halfcheetah-medium-replay-v2',
    # 'halfcheetah-medium-expert-v2',
    # 'walker2d-random-v2',
    # 'walker2d-medium-v2',
    # 'walker2d-expert-v2',
    # 'walker2d-medium-replay-v2',
    # 'walker2d-medium-expert-v2',
    # 'hopper-random-v2',
    # 'hopper-medium-v2',
    # 'hopper-expert-v2',
    # 'hopper-medium-replay-v2',
    # 'hopper-medium-expert-v2',
    # 'ant-random-v2',
    # 'ant-medium-v2',
    # 'ant-expert-v2',
    # 'ant-medium-replay-v2',
    # 'ant-medium-expert-v2',
]

ADROIT_ENVS = [
    "pen-human-v1",
    "hammer-human-v1",
    "door-human-v1",
    "relocate-human-v1",
]

FRANKA_KITCHEN = ["kitchen-complete-v0"]

ENVS = (
    MAZE2D_ENVS
    + MAZE2D_DENSE_ENVS
    + ANT_MAZE_ENVS
    + GYM_MUJOCO_ENVS
    + ADROIT_ENVS
    + FRANKA_KITCHEN
)
ENV_VARIABLES = {
    "WANDB_API_KEY": "$WANDB_API_KEY",
    "D4RL_DATASET_DIR": '"/mnt/d4rl"',
    "MUJOCO_PY_MUJOCO_PATH": '"/home/aiscuser/.mujoco/mujoco210"',
    "LD_LIBRARY_PATH": '"$$LD_LIBRARY_PATH:/home/aiscuser/.mujoco/mujoco210/bin"',
    "PYTHONPATH": '"$$PYTHONPATH:/home/aiscuser/.python_packages"',
}
SEEDS = range(1)
WANDB_OFFLINE = False
USE_WANDB = True


def generate_yaml(
    cmds,
    target_yaml,
    static_yaml_segment,
    output_yaml_path,
    per_node_commands,
    num_gpu=1,
):
    yaml = target_yaml + "\n\n" + static_yaml_segment
    yaml += "\n\njobs:"

    # environment variables to command
    if len(ENV_VARIABLES) > 1:
        env_var_cmd = " && ".join(f"export {k}={v}" for k, v in ENV_VARIABLES.items())
    else:
        env_var_cmd = None

    for start_idx in range(0, len(cmds), per_node_commands):
        cmd_batch = cmds[start_idx : start_idx + per_node_commands]

        # add job
        job_name = "_".join(_["job_name"] for _ in cmd_batch)
        job_name = hashlib.md5(job_name.encode()).hexdigest()
        yaml += (
            f"\n   - name: {job_name}" + f"\n     sku: G{num_gpu}" + f"\n     command:"
        )

        if WANDB_OFFLINE:
            yaml += f"\n     - wandb offline"

        # add environment variables
        if env_var_cmd is not None:
            yaml += f"\n     - {env_var_cmd}"

        # add python command
        python_cmd = " & ".join(_["cmd"] for _ in cmd_batch)
        yaml += f"\n     - {python_cmd}"

        if WANDB_OFFLINE:
            yaml += f"\n     - wandb sync --sync-all"

    # write yaml file
    with open(output_yaml_path, "w") as yaml_file:
        yaml_file.write(yaml)


def main():
    # generate python commands for grid-search
    cmds = []

    for env_name in ENVS:
        for seed in SEEDS:
            cmd = (
                f"python scripts/td3.py "
                + f" --seed {seed}"
                + f" --env {env_name} "
                + f" --start-time-steps 50000"
                + f" --max-time-steps 10000000"
                + f" --hidden-dim 256"
                + f" --batch-size 512"
                + f" --result-dir /mnt/opcctd3/results"
            )
            if USE_WANDB:
                cmd += f" --use-wandb"
                cmd += f" --wandb-project-name opcc-td3"
            cmds.append({"job_name": hashlib.md5(cmd.encode()).hexdigest(), "cmd": cmd})

    if len(cmds) > 0:
        if os.path.exists(os.path.join(os.getcwd(), "opcc_td3_generic.yaml")):
            os.remove(os.path.join(os.getcwd(), "opcc_td3_generic.yaml"))
        generate_yaml(
            cmds,
            GENERIC_TARGET,
            STATIC_YAML_SEGMENT,
            os.path.join(os.getcwd(), "opcc_td3_generic.yaml"),
            per_node_commands=1,
            num_gpu=1,
        )


if __name__ == "__main__":
    main()
