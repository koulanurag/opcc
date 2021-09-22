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
```python console
>>> import cque
>>> env_name = 'd4rl:maze2d-open-v0'
>>> queries = cque.get_queries(env_name)

>>> # Queries are dictonaries with policies as keys and corresponding queries as values.
>>> # Batch iteration through Queries :
>>> for (policy_a_id, policy_b_id) in queries:
        env_name_a, pre_trained_id_a = policy_a_id
        env_name_b, pre_trained_id_b = policy_b_id
        policy_a = policybazaar.get_policy(env_name_a, pre_trained_id_a)
        policy_b = policybazaar.get_policy(env_name_b, pre_trained_id_b)
        state_a, action_a, state_b, action_b, target_a, target_b, target = queries[(policy_a_id, policy_b_id)]
```

## Environments
