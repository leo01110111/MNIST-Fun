#images and labels -> list of tuples with images and labels
import struct
from array import array
import numpy as np

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
            
            labels = array("B", file.read())  

        with open(images_dir, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            
            images_raw = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0]*rows*cols)
        for i in range(size):
            img = np.array(images_raw[i*rows*cols:(i+1)*rows*cols])
            img = img.reshape(28,28)
            images[i] = img
        
        return images, labels
    
    def load(self):
        train_img, train_labels = self.read(self.training_images_dir, self.training_labels_dir)
        test_img, test_labels = self.read(self.testing_images_dir, self.testing_labels_dir)
        return (train_img, train_labels) , (test_img, test_labels)
