from source.models.basic_mlp import MLP
from source.utils.load import Loader
import json

directories = {}

with open("directories.json", "r") as reader:
    directories = json.load(reader)

loader = Loader(**directories)
training_set, testing_set = loader.load()

model = MLP()

model.train(training_set, testing_set)




