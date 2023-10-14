import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


# Initialize Network weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim, max_action=1):
        super(ActorNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = num_actions

    def forward(self, state):
        x = torch.tanh(self.linear1(state))
        x = torch.tanh(self.linear2(x))
        action_mean = torch.tanh(self.linear3(x))

        return self.max_action * action_mean
