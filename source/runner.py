from models.basic_mlp import MLP
from utils.load import Loader
import json
import os

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

model = MLP()

model.train(training_set, testing_set)

model.save_checkpoint()
print("Model Saved")




