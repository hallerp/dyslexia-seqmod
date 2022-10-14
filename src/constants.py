from scipy.stats import loguniform
import numpy as np

hyperparameter_space = {
    "cnn": {
        "batch_size": [8, 16, 32, 64, 128],
        "lr": loguniform.rvs(1e-5, 1e-1, size=15),
        "dropout": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "conv1_channels": [5, 10, 15, 20, 25, 30],
        "conv1_kernel": [3, 5],
        "pool1": ["avg", "max"],
        "conv2_channels": [10, 20, 30, 40, 50],
        "conv2_kernel": [3, 5],
        "linear1_out": [10, 20, 30, 40, 50, 60],
        "decision_boundary": loguniform.rvs(0.35, 0.65, size=20)
    },
    "lstm": {
        "batch_size": [8, 16, 32, 64, 128],
        "lr": loguniform.rvs(1e-5, 1e-1, size=15),
        "lstm_hidden_size": [10, 20, 30, 40, 50, 60, 70],
        "decision_boundary": loguniform.rvs(0.35, 0.65, size=20)
    }
}

default_params = {
    "cnn": {
        "epochs": 40,
        "batch_size": 32,
        "lr": 0.001,
        "dropout": 0.3,
        "conv1_channels": 20,
        "conv1_kernel": 3,
        "pool1": "max",
        "conv2_channels": 50,
        "conv2_kernel": 5,
        "linear1_out": 20,
        "decision_boundary": 0.5
    },
    "lstm": {
        "epochs": 40,
        "batch_size": 32,
        "lr": 0.001,
        "lstm_hidden_size": 50,
        "decision_boundary": 0.5
    }
}

features = [
    "fposx",
    "gaze",
    "firlp",
    "laslp",
    "fixdur",
    "osacdur",
    "osacdx",
    "osacdy",
    "osacl",
    "isacdur",
    "isacdx",
    "isacdy",
]
