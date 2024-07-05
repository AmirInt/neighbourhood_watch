import os
import numpy as np
import random


class Dataset:
    def __init__(self, config: dict):
        self.load_dataset(config)

    def load_dataset(self, config: dict) -> None:
        self.labels = config["dataset"]["labels"]

        if config["dataset"]["train_test_x_y_seperate"] == True:
            self.cross_validation_fold = 1
            self.cross_validation_chunk_size = 0

            if config["dataset"]["format"] == "numpy":
                self.train_x = np.load(config["dataset"]["train_x"])
                self.train_y = np.load(config["dataset"]["train_y"])
                self.test_x = np.load(config["dataset"]["test_x"])
                self.test_y = np.load(config["dataset"]["test_y"])
            
            elif config["dataset"]["format"] == "text":
                self.train_x = np.loadtxt(config["dataset"]["train_x"])
                self.train_y = np.loadtxt(config["dataset"]["train_y"])
                self.test_x = np.loadtxt(config["dataset"]["test_x"])
                self.test_y = np.loadtxt(config["dataset"]["test_y"])

        else:
            self.cross_validation_fold = config["dataset"]["cross_validation_fold"]
            y_index = config["dataset"]["y_index"]
            self.cross_validation_index = 0
            
            if config["dataset"]["format"] == "numpy":
                data = np.load(config["dataset"]["data"])
            
            elif config["dataset"]["format"] == "text":
                labels = [label.encode("ascii") for label in self.labels]
                data = np.loadtxt(config["dataset"]["data"], converters={y_index: lambda s: labels.index(s)})

            self.cross_validation_chunk_size = int(len(data) / self.cross_validation_fold)

            self.X = data[:, :y_index]
            self.y = data[:, y_index]
            

    def get_cross_validation_folds(self) -> int:
        return self.cross_validation_fold


    def get_next_dataset(self) -> dict:
        if self.cross_validation_fold != 1:
            self.test_x = self.X[self.cross_validation_index * self.cross_validation_chunk_size:(self.cross_validation_index + 1) * self.cross_validation_chunk_size]
            self.test_y = self.y[self.cross_validation_index * self.cross_validation_chunk_size:(self.cross_validation_index + 1) * self.cross_validation_chunk_size]

            self.train_x = np.append(
                self.X[0:self.cross_validation_index * self.cross_validation_chunk_size],
                self.X[(self.cross_validation_index + 1) * self.cross_validation_chunk_size:], axis=0)
            
            self.train_y = np.append(
                self.y[0:self.cross_validation_index * self.cross_validation_chunk_size],
                self.y[(self.cross_validation_index + 1) * self.cross_validation_chunk_size:], axis=0)
            
            if self.cross_validation_index < self.cross_validation_fold - 1:
                self.cross_validation_index += 1
            
        dataset = {
            "train_x": self.train_x,
            "train_y": self.train_y,
            "test_x": self.test_x,
            "test_y": self.test_y,
            "labels": self.labels
        }
        return dataset
