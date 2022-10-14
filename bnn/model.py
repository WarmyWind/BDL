import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth):
        super().__init__()
        # self.linear = nn.Linear(input_dim, output_dim)
        self.input_layer = BayesianLinear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [BayesianLinear(hidden_dim, hidden_dim) for i in range(hidden_depth)]
        )
        # self.blinear2 = BayesianLinear(hidden_dim, hidden_dim)
        self.output_layer = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x):
        x_ = self.input_layer(x)
        x_ = F.relu(x_)
        for hidden_layer in self.hidden_layers:
            x_ = hidden_layer(x_)
            x_ = F.relu(x_)
        x_ = self.output_layer(x_)
        # x_ = F.relu(x_)
        return x_


def evaluate_regression(regressor, X, y, samples=100, std_multiplier=2):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()