import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


# Initialize Network weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)
        self.linear3.weight.data.fill_(0)
        self.linear3.bias.data.fill_(0)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim, action_std):
        super(ActorNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)
        self.action_var = torch.full((num_actions,), action_std * action_std)

    def forward(self, state):
        x = torch.tanh(self.linear1(state))
        x = torch.tanh(self.linear2(x))
        action_mean = torch.tanh(self.linear3(x))

        return action_mean

    def dist(self, state):
        action_mean = self.forward(state)
        cov_mat = torch.diag(self.action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        return dist

    def to(self, device):
        self.action_var = self.action_var.to(device)
        return super(ActorNetwork, self).to(device)


class ActorCriticNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_std):
        super(ActorCriticNetwork, self).__init__()
        self.actor = ActorNetwork(num_inputs, num_actions, hidden_dim, action_std)
        self.critic = ValueNetwork(num_inputs, hidden_dim)

    def to(self, device):
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        return super(ActorCriticNetwork, self).to(device)
