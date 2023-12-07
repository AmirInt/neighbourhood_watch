import numpy as np
import matplotlib.pyplot as plt
import time


def visualise_image_vector(vector: np.ndarray, image_shape: tuple):
    plt.axis('off')
    plt.imshow(vector.reshape(image_shape), cmap=plt.cm.gray)
    plt.show()




class NearestNeighbour:
    def __init__(self, train_data: np.ndarray, train_labels: np.ndarray):
        self._train_data = train_data
        self._train_labels = train_labels
        self._dist_func = self.l2_distance
    

    def l1_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.linalg.norm(x=(x - y), ord=1)


    def l2_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.linalg.norm(x - y)


    def change_dist_func(self, distance_function: str = "l2") -> None:
        match distance_function:
            case "l1":
                self._dist_func = self.l1_distance
            case "l2":
                self._dist_func = self.l2_distance
            case other:
                self._dist_func = self.l2_distance


    def find_nearest_neighbour_index(self, x: np.ndarray) -> int:
        if x.shape[0] != self._train_data.shape[1]:
            raise ValueError

        dists = [self._dist_func(x, data) for data in self._train_data]

        return np.argmin(dists)


    def __call__(self, y: np.ndarray):
        index = self.find_nearest_neighbour_index(y)

        return self._train_labels[index]