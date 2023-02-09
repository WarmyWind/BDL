import torch
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Dropout
import gpytorch
import numpy as np


class GraphDKL(gpytorch.Module):
    def __init__(self, feature_extractor, gp):
        """
        This wrapper class is necessary because ApproximateGP (above) does some magic
        on the forward method which is not compatible with a feature_extractor.
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.gp = gp

    def forward(self, data):
        # x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        features = self.feature_extractor(data)
        # features = torch.reshape(data.num_graph, -1, )
        return self.gp(features)

    def detailed_loss(self, data, criterion, reduction='sum', labels=None, K=None, complexity_cost_weight=1):
        y_pred = self(data)
        y_mean = y_pred.mean
        y = torch.nn.functional.sigmoid(y_mean)
        if labels != None:
            likelihood_cost = criterion(y, labels)
        else:
            likelihood_cost = criterion(data, y, K, reduction)
        # std = y_pred.stddev
        loss = likelihood_cost + complexity_cost_weight * self.gp.variational_strategy.kl_divergence()
        return y_pred, likelihood_cost.cpu().detach().numpy(), loss.cpu().detach().numpy()