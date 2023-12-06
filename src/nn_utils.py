import numpy as np
import matplotlib.pyplot as plt
import time


def visualise_image_vector(vector: np.ndarray, image_shape: tuple):
    plt.axis('off')
    plt.imshow(vector.reshape(image_shape), cmap=plt.cm.gray)
    plt.show()




class NearestNeighbour:
    def __init__(self, train_dataset: np.ndarray, train_labels: np.ndarray):
        self._train_dataset = train_dataset
        self._train_labels = train_labels
    

    def square_eucl_dist(x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(np.square(x - y))


    def find_nearest_neighbour_index(self, x: np.ndarray) -> int:
        if x.shape[0] != self._train_dataset.shape[1]:
            raise ValueError
        
        dists = [square_eucl_dist(x, data) for data in self._train_dataset]

        return np.argmin(dists)


    def __call__(self, y: np.ndarray):
        index = self.find_nearest_neighbour_index(y)

        return self._train_labels[index]