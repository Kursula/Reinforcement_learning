import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        
        dim = 300
        
        # Actor 
        self.actor1 = nn.Linear(state_dim, dim)
        self.actor2 = nn.Linear(dim, dim)
        self.actor3 = nn.Linear(dim, action_dim)

        # Critic
        self.critic1 = nn.Linear(state_dim, dim)
        self.critic2 = nn.Linear(dim, dim)
        self.critic3 = nn.Linear(dim, 1)


    def forward(self, x):
        actor = self.actor1(x)
        actor = torch.tanh(actor)
        actor = self.actor2(actor)
        actor = torch.tanh(actor)
        logits = self.actor3(actor)

        critic = self.critic1(x)
        critic = torch.tanh(critic)
        critic = self.critic2(critic)
        critic = torch.tanh(critic)
        values = self.critic3(critic)

        return logits, values