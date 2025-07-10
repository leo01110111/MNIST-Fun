import os
import datetime
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
    def __init__(self, alpha=0.001, output_dims = 10, input_dims = 784,
            l1_dims=256, l2_dims=256,l3_dims=256, chkpt_dir='checkpoints'):
        super(MLP, self).__init__()

        parent_dir = os.path.dirname(os.path.abspath(__file__))
        chkpt_dir  = os.path.join(parent_dir, chkpt_dir)
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(chkpt_dir, exist_ok=True)

        now =  datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H%M%S")

        self.checkpoint_file = os.path.join(chkpt_dir, f'mlp-{now}')
        self.network = nn.Sequential(
                nn.Linear(input_dims, l1_dims),
                nn.ReLU(),
                nn.Linear(l1_dims, l2_dims),
                nn.ReLU(),
                nn.Linear(l2_dims, l3_dims),
                nn.ReLU(),
                nn.Linear(l3_dims, output_dims)
                # Removed Softmax - CrossEntropyLoss applies it internally
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

    def train(self, training_dataset, testing_dataset, batch_size=60,
              epoch=4):
        self.batch_size = batch_size
        
        # Flatten the images from (N, 28, 28) to (N, 784)
        train_images = training_dataset[0].view(training_dataset[0].size(0), -1)
        test_images = testing_dataset[0].view(testing_dataset[0].size(0), -1)
        
        train_dataset = torch.utils.data.TensorDataset(train_images, training_dataset[1])
        test_dataset = torch.utils.data.TensorDataset(test_images, testing_dataset[1])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle=False)

        criteria = nn.CrossEntropyLoss() #define loss function.

        for i in range(epoch):

            self.network.train()
            training_loss = 0
            testing_loss = 0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                prediction = self.network(images)
                loss = criteria(prediction, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                training_loss += loss.item()
            
            self.network.eval()  # Set to evaluation mode for testing
            with torch.no_grad():  # Disable gradients fo testing
                for images, labels in self.test_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    prediction = self.network(images)
                    loss = criteria(prediction, labels)
                    testing_loss += loss.item()
           
            # Calculate average losses
            avg_train_loss = training_loss / len(train_loader)
            avg_test_loss = testing_loss / len(self.test_loader)
            
            print(f"Epoch {i}, Avg Training Loss: {avg_train_loss:.4f}")
            print(f"Epoch {i}, Avg Testing Loss: {avg_test_loss:.4f}")
        print("Training Complete")
        
