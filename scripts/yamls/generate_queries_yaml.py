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
    - pip3 install absl-py==1.0.0
    - pip3 install numpy==1.21.5
    - pip3 install scikit-learn
    - pip3 install gym==0.21.0
    - pip3 install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
    - pip3 install "cython<3" --upgrade
    - echo 'export CPATH="/usr/include:$$CPATH"' >> /home/aiscuser/.bashrc
    - chmod -R 777 /opt/conda/lib/python3.8/site-packages/mujoco_py
    - pip3 install pandas==1.3.5
    - pip3 install plotly==5.5.0
    - pip3 install moviepy
    - pip3 install plotly

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

ADROIT_ENVS = [
    "d4rl:pen-v0",
    "d4rl:hammer-v0",
    "d4rl:door-v0",
]

ENVS = ADROIT_ENVS
HORIZONS = [10, 20, 30, 40, 50]
IGNORE_DELTAS_PER_HORIZON = {
    "d4rl:pen-v0": [700, 1400, 2100, 2800, 3500],
    "d4rl:hammer-v0": [1000, 2000, 3000, 4000, 5000],
    "d4rl:door-v0": [200, 400, 600, 800, 1000],
}

ENV_VARIABLES = {
    "WANDB_API_KEY": "$WANDB_API_KEY",
    "D4RL_DATASET_DIR": '"/mnt/d4rl"',
    "MUJOCO_PY_MUJOCO_PATH": '"/home/aiscuser/.mujoco/mujoco210"',
    "LD_LIBRARY_PATH": '"$$LD_LIBRARY_PATH:/home/aiscuser/.mujoco/mujoco210/bin"',
    "PYTHONPATH": '"$$PYTHONPATH:/home/aiscuser/.python_packages"',
}
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
            yaml += f"\n       - {env_var_cmd}"

        yaml += f"\n       - pip install -e ."
        yaml += f"\n       - python -c \"import os; [os.rename('/scratch/amlt_code/opcc/assets/'+x, '/scratch/amlt_code/opcc/assets/'+x.replace('%3A',':')) for x in os.listdir('/scratch/amlt_code/opcc/assets')]\""
        yaml += f"\n       - pip install torch --upgrade"

        # add python command
        python_cmd = " & ".join(_["cmd"] for _ in cmd_batch)
        yaml += f"\n       - {python_cmd}"

        if WANDB_OFFLINE:
            yaml += f"\n     - wandb sync --sync-all"

    # write yaml file
    with open(output_yaml_path, "w") as yaml_file:
        yaml_file.write(yaml)


def main():
    # generate python commands for grid-search
    cmds = []

    for env_name in ENVS:
        cmd = (
            f"python "
            "scripts/generate_queries.py"
            f" --env-name {env_name}"
            f" --horizons {' '.join([str(x) for x in HORIZONS])}"
            " --policy-ids 1 2 3 4"
            " --noise 0.1 "
            " --eval-runs 10"
            f" --ignore-delta-per-horizons {' '.join([str(x) for x in IGNORE_DELTAS_PER_HORIZON[env_name]])}"
            " --max-trans-count 10000 "
            "--ignore-stuck-count 500"
            " --save-prob 0.6"
            " --per-policy-comb-query 250"
        )

        if USE_WANDB:
            cmd += f" --use-wandb"

        cmds.append({"job_name": hashlib.md5(cmd.encode()).hexdigest(), "cmd": cmd})

    if len(cmds) > 0:
        if os.path.exists(os.path.join(os.getcwd(), "opcc_queries_generic.yaml")):
            os.remove(os.path.join(os.getcwd(), "opcc_queries_generic.yaml"))
        generate_yaml(
            cmds,
            GENERIC_TARGET,
            STATIC_YAML_SEGMENT,
            os.path.join(os.getcwd(), "opcc_queries_generic.yaml"),
            per_node_commands=1,
            num_gpu=1,
        )


if __name__ == "__main__":
    main()
