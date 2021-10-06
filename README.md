# Contrastive Queries (cque)

It's a benchmark comprising of queries to evaluate uncertainty estimation in offline reinforcement learning.

## Installation
It requires:

- [Python 3.6+](https://www.python.org/downloads/)
- [mujoco-py](https://github.com/openai/mujoco-py), [mujoco 200](https://www.roboti.us/index.html) and [mujoco license](https://www.roboti.us/license.html). Please, follow `mujoco-py` installation instructions from [here](https://github.com/openai/mujoco-py).
- [Pytorch >= 1.8.0](https://pytorch.org/)

Python package and dependencies could be installed using:
```bash
pip install git+https://github.com/koulanurag/cque@main#egg=cque
```
Or
```bash
git clone https://github.com/koulanurag/cque.git
cd cque
pip install -e .
```

## Usage

```python
import cque, policybazaar

env_name = 'd4rl:maze2d-open-v0'
dataset_name = '1k'

# Queries are dictonaries with policies as keys and corresponding queries as values.
# Batch iteration through Queries :
queries = cque.get_queries(env_name)
for (policy_a_id, policy_b_id), query_batch in queries.items():
    policy_a = policybazaar.get_policy(*policy_a_id)
    policy_b = policybazaar.get_policy(*policy_b_id)

    state_a, action_a, horizon_a, state_b, action_b, horizon_b, target_a, target_b, target = query_batch
    
# Datasets:
# This is a very-slim wrapper over D4RL datasets
dataset = cque.get_dataset(env_name, dataset_name)

``` 

## Environments
- We borrow dataset's from D4RL 
- Queries data can be visualized [**here**](https://wandb.ai/koulanurag/cque/reports/Visualization-of-Queries--VmlldzoxMDkxMjcx).

### :low_brightness: [d4rl:maze2d](https://github.com/rail-berkeley/d4rl/wiki/Tasks#maze2d)

| Environment Name | Datasets |
|:------: | :------: | 
|`d4rl:maze2d-open-v0`|`1k, 10k, 100k`|
|`d4rl:maze2d-medium-v1`|`1k, 10k, 100k`|
|`d4rl:maze2d-umaze-v1`|`1k, 10k, 100k`|
|`d4rl:maze2d-large-v1`|`1k, 10k, 100k`|

### :low_brightness: [mujoco(gym)](https://gym.openai.com/envs/#mujoco)

| Environment Name | Datasets|
|:------: |:------:|
|`HalfCheetah-v2`| `random, expert, medium, medium-replay, medium-expert`|
|`Hopper-v2`| `random, expert, medium, medium-replay, medium-expert`|
|`Walker2d-v2`| `random, expert, medium, medium-replay, medium-expert`|