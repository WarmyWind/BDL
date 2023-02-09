from datetime import datetime
import json
import pathlib
import random
from pathlib import Path
import torch
import numpy as np

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def one_hot_embedding(labels, num_classes, device=None):
    # Convert to One Hot Encoding
    if device == None:
        device = get_device()
    y = torch.eye(num_classes)
    y = y.to(device)
    return y[labels.flatten().type(torch.long)]

def set_seed(seed):
    if seed is None:
        seed = random.randint(0, 1000)

    np.random.seed(seed)
    torch.manual_seed(seed)

    return seed


def get_results_directory(name, stamp=True):
    timestamp = datetime.now().strftime("%Y-%m-%d-%A-%H-%M-%S")

    results_dir = pathlib.Path("runs")

    if name is not None:
        results_dir = results_dir / name

    results_dir = results_dir / timestamp if stamp else results_dir

    results_dir.mkdir(parents=True)

    return results_dir


class Hyperparameters:
    def __init__(self, *args, **kwargs):
        """
        Optionally pass a Path object to load hypers from

        If additional values are passed they overwrite the loaded ones
        """
        if len(args) == 1:
            self.load(args[0])

        self.from_dict(kwargs)

    def to_dict(self):
        return vars(self)

    def from_dict(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)

    def save(self, path):
        # path.write_text(self.to_json())
        p = Path(path)
        p.write_text(self.to_json())


    def load(self, path):
        self.from_dict(json.loads(path.read_text()))

    def __contains__(self, k):
        return hasattr(self, k)

    def __str__(self):
        return f"Hyperparameters:\n {self.to_json()}"
