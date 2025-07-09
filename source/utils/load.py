#images and labels -> list of tuples with images and labels
import struct
from array import array
import numpy as np
import torch

class Loader():
    def __init__(self, training_images_dir, training_labels_dir, testing_images_dir, testing_labels_dir):
        self.training_images_dir = training_images_dir
        self.training_labels_dir = training_labels_dir
        self.testing_images_dir = testing_images_dir
        self.testing_labels_dir = testing_labels_dir
    
    def read(self, images_dir, labels_dir):
        with open(labels_dir, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            
            labels = np.frombuffer(file.read(), np.uint8, offset=8)

        with open(images_dir, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            
            images_raw = file.read()
            print(images_raw)
            images_raw = np.frombuffer(file.read(), np.uint8, offset=16)

        images = np.array()
        for i in range(size):
            images.append([0]*rows*cols)
        for i in range(size):
            img = np.array(images_raw[i*rows*cols:(i+1)*rows*cols])
            img = img.reshape(28,28)
            images[i] = img
        
        return images, labels
    
    def load(self, batch = 60):
        train_img, train_labels = self.read(self.training_images_dir, self.training_labels_dir)
        test_img, test_labels = self.read(self.testing_images_dir, self.testing_labels_dir)
        train_img = torch.Tensor(train_img / 255.0, dtype=torch.float32) #0-255 -> 0-1
        test_img = torch.Tensor(test_img / 255.0, dtype=torch.float32)
        return (train_img, train_labels), (test_img, test_labels)
