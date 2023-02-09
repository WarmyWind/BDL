import torch
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Sigmoid, Dropout
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator
import numpy as np


class IGConv(MessagePassing):
    def __init__(self, mlp1, mlp2, **kwargs):
        super(IGConv, self).__init__(aggr='max', **kwargs)

        self.mlp1 = mlp1
        self.mlp2 = mlp2
        # self.reset_parameters()

    # def reset_parameters(self):
    #     reset(self.mlp1)
    #     reset(self.mlp2)

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=1)
        comb = self.mlp2(tmp)
        return torch.cat([x[:, :2], comb], dim=1)

    def forward(self, x, edge_index, edge_attr):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_j, edge_attr], dim=1)
        agg = self.mlp1(tmp)
        return agg

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.mlp1, self.mlp2)


def MLP(channels, dropout_rate=0.0):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i], bias=True), Dropout(p=dropout_rate), ReLU())
        for i in range(1, len(channels))
    ])


def BNN(channels):
    return Seq(*[
        Seq(BayesianLinear(channels[i - 1], channels[i], bias=True), ReLU())
        for i in range(1, len(channels))
    ])


class IGCNet(torch.nn.Module):
    def __init__(self):
        super(IGCNet, self).__init__()

        self.mlp1 = MLP([5, 16, 32])
        self.mlp2 = MLP([35, 16])
        self.mlp2 = Seq(*[self.mlp2, Seq(Lin(16, 1, bias=True), Sigmoid())])
        self.conv = IGConv(self.mlp1, self.mlp2)

    def forward(self, data):
        x, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        # x3 = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        # x4 = self.conv(x = x3, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return out

class MCDropoutIGCNet(torch.nn.Module):
    def __init__(self):
        super(MCDropoutIGCNet, self).__init__()

        self.mlp1 = MLP([5, 16, 32], dropout_rate=0.2)
        self.mlp2 = MLP([35, 16], dropout_rate=0.2)
        self.mlp2 = Seq(*[self.mlp2, Seq(Lin(16, 1, bias=True), Sigmoid())])
        self.conv = IGConv(self.mlp1, self.mlp2)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
        x2 = self.conv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        # x3 = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        # x4 = self.conv(x = x3, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        return out

    def sample_detailed_loss(self,
                             inputs,
                             criterion,
                             sample_nbr,
                             reduction='sum',
                             labels=None,
                             K=None):

        loss = []
        y_hat = []
        for _ in range(sample_nbr):
            outputs = self(inputs)
            y_hat.append(outputs.cpu().detach().numpy())
            if labels != None:
                loss.append(criterion(outputs, labels).cpu().detach().numpy())
            else:
                loss.append(criterion(inputs, outputs, K, reduction).cpu().detach().numpy())

        if reduction == 'none':
            return np.array(y_hat), np.array(loss)
        else:
            return np.array(y_hat), np.mean(loss)


@variational_estimator
class BayesianIGCNet(torch.nn.Module):
    def __init__(self):
        super(BayesianIGCNet, self).__init__()

        self.BNN1 = BNN([5, 16, 32])
        self.BNN2 = BNN([35, 16])
        self.BNN2 = Seq(*[self.BNN2, Seq(BayesianLinear(16, 1, bias=True), Sigmoid())])
        self.conv = IGConv(self.BNN1, self.BNN2)

    def forward(self, data):
        x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
        x1 = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
        x2 = self.conv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
        # x3 = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
        # x4 = self.conv(x = x3, edge_index = edge_index, edge_attr = edge_attr)
        out = self.conv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
        return out

    def my_sample_elbo(self, inputs, criterion, sample_nbr, complexity_cost_weight=1, labels=None, K=None):
        loss = 0
        for _ in range(sample_nbr):
            outputs = self(inputs)
            if labels != None:
                loss += criterion(outputs, labels)
            else:
                loss += criterion(inputs, outputs, K)
            loss += self.nn_kl_divergence() * complexity_cost_weight
        return loss / sample_nbr

    def my_sample_elbo_detailed_loss(self,
                                     inputs,
                                     criterion,
                                     sample_nbr,
                                     reduction='sum',
                                     complexity_cost_weight=1,
                                     labels=None,
                                     K=None):
        #     loss = 0
        #     likelihood_cost = 0
        #     complexity_cost = 0
        #     y_hat = []
        #     for _ in range(sample_nbr):
        #         outputs = self(inputs)
        #         y_hat.append(outputs.cpu().detach().numpy())
        #         if labels != None:
        #             likelihood_cost += criterion(outputs, labels)
        #             complexity_cost += self.nn_kl_divergence() * complexity_cost_weight
        #         else:
        #             likelihood_cost += criterion(inputs, outputs, K, reduction)
        #             complexity_cost += self.nn_kl_divergence() * complexity_cost_weight
        #         loss += self.nn_kl_divergence() * complexity_cost_weight

        # loss = []
        likelihood_cost = []
        complexity_cost = []
        y_hat = []
        for _ in range(sample_nbr):
            outputs = self(inputs)
            y_hat.append(outputs.cpu().detach().numpy())
            if labels != None:
                likelihood_cost.append(criterion(outputs, labels).cpu().detach().numpy())
                complexity_cost.append(self.nn_kl_divergence().cpu().detach().numpy() * complexity_cost_weight)
            else:
                likelihood_cost.append(criterion(inputs, outputs, K, reduction).cpu().detach().numpy())
                complexity_cost.append(self.nn_kl_divergence().cpu().detach().numpy() * complexity_cost_weight)
            # loss.append(self.nn_kl_divergence().cpu().detach().numpy() * complexity_cost_weight)

        loss = np.array(likelihood_cost) + np.array(complexity_cost)[:,np.newaxis] * np.ones(np.array(likelihood_cost).shape)
        if reduction == 'none':
            return np.array(y_hat), loss, \
                   np.array(likelihood_cost), \
                   np.array(complexity_cost)
        else:
            return np.array(y_hat), loss, \
                   np.mean(likelihood_cost), \
                   np.mean(complexity_cost)


# class IGC(torch.nn.Module):
#     def __init__(self):
#         super(IGC, self).__init__()
#
#         self.mlp1 = MLP([5, 16, 32])
#         self.mlp2 = MLP([35, 16])
#         self.mlp2 = Seq(*[self.mlp2, Seq(Lin(16, 1, bias=True), Sigmoid())])
#         self.conv = IGConv(self.mlp1, self.mlp2)
#
#     def forward(self, data):
#         x0, edge_attr, edge_index = data.x, data.edge_attr, data.edge_index
#         x1 = self.conv(x=x0, edge_index=edge_index, edge_attr=edge_attr)
#         x2 = self.conv(x=x1, edge_index=edge_index, edge_attr=edge_attr)
#         # x3 = self.conv(x = x2, edge_index = edge_index, edge_attr = edge_attr)
#         # x4 = self.conv(x = x3, edge_index = edge_index, edge_attr = edge_attr)
#         out = self.conv(x=x2, edge_index=edge_index, edge_attr=edge_attr)
#         return out