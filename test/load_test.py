import json
from source.utils import load
import matplotlib.pyplot as plt

with open("directories.json", "r") as reader:
    directories = json.load(reader)

data = load.Loader(**directories)

(t_images, t_labels), (v_images, v_labels) = data.load()

variables = [t_images, t_labels, v_images, v_labels]

#prints out the size of the dataset
for i in variables:
    print(f"size of this thing is {len(i)}")


# Display the first training image
plt.imshow(t_images[0], cmap='gray')
plt.title(f"Label: {t_labels[0]}")
plt.show()