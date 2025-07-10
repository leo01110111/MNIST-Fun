from models.basic_mlp import MLP
from utils.load import Loader
import json
import os
import torch

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (MNIST-Fun)
project_root = os.path.dirname(script_dir)

directories = {}

with open(os.path.join(project_root, "directories.JSON"), "r") as reader:
    directories = json.load(reader)

# Update all directory paths to be absolute paths
for key, path in directories.items():
    directories[key] = os.path.join(project_root, path)

print((directories))

loader = Loader(**directories)
training_set, testing_set = loader.load()

model = MLP(alpha=0.001)  # Lower learning rate

model.train(training_set, testing_set, batch_size=32, epoch=10)  # Smaller batch size, fewer epochs

model.save_checkpoint()
print("Model Saved")

# Test the model
correct = 0
total = 0

model.network.eval()  # Set to evaluation mode
with torch.no_grad():  # Disable gradient computation for testing
    for images, labels in model.test_loader:
        images, labels = images.to(model.device), labels.to(model.device)
        
        # Get predictions
        outputs = model.network(images)
        _, predicted = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")








