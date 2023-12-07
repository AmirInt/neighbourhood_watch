import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.neighbors import BallTree, KDTree


def visualise_image_vector(vector: np.ndarray, image_shape: tuple):
    plt.axis('off')
    plt.imshow(vector.reshape(image_shape), cmap=plt.cm.gray)
    plt.show()




class NearestNeighbour:
    def __init__(self, train_data: np.ndarray, train_labels: np.ndarray, preprocess_method: str):
        self._train_data = train_data
        self._train_labels = train_labels
        self._dist_func = self.l2_distance

        self._preprocess_method = preprocess_method

        match preprocess_method:
            case "ball_tree":
                self._ball_tree = BallTree(self._train_data)
            case "kd_tree":
                self._kd_tree = KDTree(train_data)
            case other:
                pass
    

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
        dists = [self._dist_func(x, data) for data in self._train_data]

        return np.argmin(dists)


    def __call__(self, y: np.ndarray):
        index = self.find_nearest_neighbour_index(y)

        return self._train_labels[index]
    

    def mass_predict(self, test_data: np.ndarray):
        match self._preprocess_method:
            case "ball_tree":
                test_neighbors = np.squeeze(self._ball_tree.query(test_data, k=1, return_distance=False))
                test_predictions = self._train_labels[test_neighbors]
            case "kd_tree":
                test_neighbors = np.squeeze(self._kd_tree.query(test_data, k=1, return_distance=False))
                test_predictions = self._train_labels[test_neighbors]
            case other:
                test_predictions = [self(test_data[i]) for i in range(len(test_data))]
        
        return test_predictions
