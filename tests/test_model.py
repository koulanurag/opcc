import gym
import pytest
import torch

import opcc
from opcc.config import ENV_CONFIGS
from opcc.config import MAX_PRE_TRAINED_LEVEL, MIN_PRE_TRAINED_LEVEL


@pytest.mark.parametrize('env_name, pre_trained',
                         [(env_name, pre_trained)
                          for env_name in list(ENV_CONFIGS.keys())
                          for pre_trained in range(MIN_PRE_TRAINED_LEVEL,
                                                   MAX_PRE_TRAINED_LEVEL)])
def test_model_exists(env_name, pre_trained):
    model, model_info = opcc.get_policy(env_name, pre_trained)

    env = gym.make(env_name)
    obs = env.reset()
    obs = torch.tensor(obs).unsqueeze(0)
    action = model.actor(obs).data.cpu().numpy()[0].astype('float32')
    critic = model.critic(obs).data.cpu().item()

    env.step(action)
    env.close()
