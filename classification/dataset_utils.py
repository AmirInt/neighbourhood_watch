import os
import numpy as np
import yaml
import random


class Dataset:
    def __init__(self):
        self.load_dataset()

    def load_config(self) -> dict:
        with open("conf.yaml") as conf_file:
            try:
                config = yaml.safe_load(conf_file)
            except yaml.YAMLError as exc:
                print(exc)
                exit()

        return config

    def load_dataset(self) -> None:
        config = self.load_config()

        self._labels = config["dataset"]["labels"]

        if config["dataset"]["train_test_x_y_seperate"] == True:
            self._cross_validation_fold = 1
            self._cross_validation_chunk_size = 0

            if config["dataset"]["format"] == "numpy":
                self._train_x = np.load(config["dataset"]["train_x"])
                self._train_y = np.load(config["dataset"]["train_y"])
                self._test_x = np.load(config["dataset"]["test_x"])
                self._test_y = np.load(config["dataset"]["test_y"])
            
            elif config["dataset"]["format"] == "text":
                self._train_x = np.loadtxt(config["dataset"]["train_x"])
                self._train_y = np.loadtxt(config["dataset"]["train_y"])
                self._test_x = np.loadtxt(config["dataset"]["test_x"])
                self._test_y = np.loadtxt(config["dataset"]["test_y"])

        else:
            self._cross_validation_fold = config["dataset"]["cross_validation_fold"]
            y_index = config["dataset"]["y_index"]
            self._cross_validation_index = 0
            
            if config["dataset"]["format"] == "numpy":
                data = np.load(config["dataset"]["data"])
            
            elif config["dataset"]["format"] == "text":
                labels = [label.encode("ascii") for label in self._labels]
                data = np.loadtxt(config["dataset"]["data"], converters={y_index: lambda s: labels.index(s)})

            self._cross_validation_chunk_size = int(len(data) / self._cross_validation_fold)

            self._X = data[:, :y_index]
            self._y = data[:, y_index]
            

    def get_cross_validation_folds(self) -> int:
        return self._cross_validation_fold


    def get_next_dataset(self) -> dict:
        if self._cross_validation_fold != 1:
            self._test_x = self._X[self._cross_validation_index * self._cross_validation_chunk_size:(self._cross_validation_index + 1) * self._cross_validation_chunk_size]
            self._test_y = self._y[self._cross_validation_index * self._cross_validation_chunk_size:(self._cross_validation_index + 1) * self._cross_validation_chunk_size]

            self._train_x = np.append(
                self._X[0:self._cross_validation_index * self._cross_validation_chunk_size],
                self._X[(self._cross_validation_index + 1) * self._cross_validation_chunk_size:], axis=0)
            
            self._train_y = np.append(
                self._y[0:self._cross_validation_index * self._cross_validation_chunk_size],
                self._y[(self._cross_validation_index + 1) * self._cross_validation_chunk_size:], axis=0)
            
            if self._cross_validation_index < self._cross_validation_fold - 1:
                self._cross_validation_index += 1
            
        dataset = {
            "train_x": self._train_x,
            "train_y": self._train_y,
            "test_x": self._test_x,
            "test_y": self._test_y,
            "labels": self._labels
        }
        return dataset
