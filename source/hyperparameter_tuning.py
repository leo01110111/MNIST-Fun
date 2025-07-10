from models.basic_mlp import MLP
from utils.load import Loader
import json
import os
import torch
import random
import itertools

def load_data():
    """Load MNIST data"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    directories = {}
    with open(os.path.join(project_root, "directories.JSON"), "r") as reader:
        directories = json.load(reader)
    
    for key, path in directories.items():
        directories[key] = os.path.join(project_root, path)
    
    loader = Loader(**directories)
    return loader.load()

def evaluate_model(model):
    """Evaluate model accuracy"""
    correct = 0
    total = 0
    
    model.network.eval()
    with torch.no_grad():
        for images, labels in model.test_loader:
            images, labels = images.to(model.device), labels.to(model.device)
            outputs = model.network(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def grid_search():
    """Grid search hyperparameter tuning"""
    training_set, testing_set = load_data()
    
    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [16, 32, 64],
        'hidden_dims': [128, 256],
        'epochs': [5, 10]
    }
    
    best_accuracy = 0
    best_params = None
    
    total_combinations = len(list(itertools.product(*param_grid.values())))
    print(f"Testing {total_combinations} combinations...")
    
    for i, (lr, bs, hd, ep) in enumerate(itertools.product(*param_grid.values())):
        print(f"\nTrial {i+1}/{total_combinations}")
        print(f"Params: lr={lr}, batch_size={bs}, hidden_dims={hd}, epochs={ep}")
        
        model = MLP(alpha=lr, l1_dims=hd, l2_dims=hd)
        model.train(training_set, testing_set, batch_size=bs, epoch=ep)
        
        accuracy = evaluate_model(model)
        print(f"Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'lr': lr, 'batch_size': bs, 'hidden_dims': hd, 'epochs': ep}
            print(f"New best! {best_params}")
    
    return best_params, best_accuracy

def random_search(n_trials=20):
    """Random search hyperparameter tuning"""
    training_set, testing_set = load_data()
    
    best_accuracy = 0
    best_params = None
    
    for trial in range(n_trials):
        # Random sampling
        lr = random.choice([0.0001, 0.001, 0.01, 0.1])
        batch_size = random.choice([16, 32, 64])
        hidden_dims = random.choice([64, 128, 256, 512])
        epochs = random.randint(5, 15)
        
        print(f"\nTrial {trial+1}/{n_trials}")
        print(f"Params: lr={lr}, batch_size={batch_size}, hidden_dims={hidden_dims}, epochs={epochs}")
        
        model = MLP(alpha=lr, l1_dims=hidden_dims, l2_dims=hidden_dims)
        model.train(training_set, testing_set, batch_size=batch_size, epoch=epochs)
        
        accuracy = evaluate_model(model)
        print(f"Accuracy: {accuracy:.4f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'lr': lr, 'batch_size': batch_size, 'hidden_dims': hidden_dims, 'epochs': epochs}
            print(f"New best! {best_params}")
    
    return best_params, best_params

if __name__ == "__main__":
    print("Starting hyperparameter tuning...")
    
    # Choose your method
    print("\n=== Random Search ===")
    best_params, best_accuracy = random_search(n_trials=10)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")
