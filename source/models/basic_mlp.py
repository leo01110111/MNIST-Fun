import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

#initalize the network
#fit the data
#print the training curve

class MLP(nn.Module):
    def __init__(self, alpha=0.5, output_dims = 10, input_dims = 784,
            fc1_dims=256, fc2_dims=256, chkpt_dir='checkpoints/MLP/'):
        super(MLP, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'mlp')
        self.network = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, output_dims),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.network(state)
        dist = Categorical(dist) # turns a probability distribution into categories of discrete states rather than a continuous value. Think about a dice, which has 6 discrete categories
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))