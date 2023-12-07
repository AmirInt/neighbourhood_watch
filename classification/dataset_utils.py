import os
import numpy as np
import yaml
import random


def load_config() -> dict:
    with open("conf.yaml") as conf_file:
        try:
            config = yaml.safe_load(conf_file)
        except yaml.YAMLError as exc:
            print(exc)
            exit()

    return config

def load_dataset() -> dict:
    config = load_config()

    labels = config["dataset"]["labels"]

    if config["dataset"]["train_test_x_y_seperate"] == True:
        if config["dataset"]["format"] == "numpy":
            train_x = np.load(config["dataset"]["train_x"])
            train_y = np.load(config["dataset"]["train_y"])
            test_x = np.load(config["dataset"]["test_x"])
            test_y = np.load(config["dataset"]["test_y"])
        
        elif config["dataset"]["format"] == "text":
            train_x = np.loadtxt(config["dataset"]["train_x"])
            train_y = np.loadtxt(config["dataset"]["train_y"])
            test_x = np.loadtxt(config["dataset"]["test_x"])
            test_y = np.loadtxt(config["dataset"]["test_y"])

    else:
        
        train_ratio = config["dataset"]["train_ratio"]
        y_index = config["dataset"]["y_index"]
        
        if config["dataset"]["format"] == "numpy":
            data = np.load(config["dataset"]["data"])
        
        elif config["dataset"]["format"] == "text":
            data = np.loadtxt(config["dataset"]["data"], converters={y_index: lambda s: labels.index(s)})

        X = data[:, :y_index]
        y = data[:, y_index]
        
        train_indices = random.sample(range(len(X)), train_ratio * len(X))

        train_x = X[train_indices]
        train_y = y[train_indices]

        test_x = np.delete(X, train_indices)
        test_y = np.delete(y, train_indices)
    
    dataset = {
        "train_x": train_x,
        "train_y": train_y,
        "test_x": test_x,
        "test_y": test_y,
        "labels": labels
    }

    return dataset