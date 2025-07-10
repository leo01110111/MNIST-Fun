import os
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

#initalize the network
#fit the data
#print the training curve

class MLP(nn.Module):
    def __init__(self, alpha=0.5, output_dims = 10, input_dims = 784,
            l1_dims=256, l2_dims=256, chkpt_dir='checkpoints/MLP/'):
        super(MLP, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'mlp')
        self.network = nn.Sequential(
                nn.Linear(input_dims, l1_dims),
                nn.ReLU(),
                nn.Linear(l1_dims, l2_dims),
                nn.ReLU(),
                nn.Linear(l2_dims, output_dims),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def inference(self, state):
        dist = self.network(state)
        dist = Categorical(dist) # turns a probability distribution into categories of discrete states rather than a continuous value. Think about a dice, which has 6 discrete categories
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    def train(self, training_dataset, testing_dataset, batch_size=60,
              epoch=4):
        
        train_dataset = torch.utils.data.TensorDataset(training_dataset[0], training_dataset[1])
        test_dataset = torch.utils.data.TensorDataset(testing_dataset[0], testing_dataset[1])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

        criteria = nn.CrossEntropyLoss() #define loss function.

        for i in range(epoch):

            self.network.train()
            training_loss = 0
            testing_loss = 0

            for images, labels in train_loader:
                prediction = self.network(images)
                loss = criteria(prediction, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                training_loss += loss.item()
            
            for images, labels in test_loader:
                prediction = self.network(images)
                loss = criteria(prediction, labels)
                
                testing_loss += loss.item()
           
            print(f"Epoch {i}, Training Loss: {training_loss}")
            print(f"Epoch {i}, Testing Loss: {testing_loss}") #does an epoch have to mean a run thru all data points or could it also mean a run thru of a batch
        print("Training Complete")
        
