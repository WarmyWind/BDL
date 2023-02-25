import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator
from lib.utils import get_device, one_hot_embedding

from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, criterion, hetero_noise_est, only_output_layer):
        super().__init__()
        self.criterion = criterion
        self.hetero_noise_est = hetero_noise_est
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.linear = nn.Linear(input_dim, output_dim)
        if not only_output_layer:
            self.input_layer = BayesianLinear(input_dim, hidden_dim)
            self.hidden_layers = nn.ModuleList(
                [BayesianLinear(hidden_dim, hidden_dim) for i in range(hidden_depth)]
            )
        else:
            self.input_layer = nn.Linear(input_dim, hidden_dim)
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for i in range(hidden_depth)]
            )
        # self.blinear2 = BayesianLinear(hidden_dim, hidden_dim)
        if hetero_noise_est:
            self.output_layer = BayesianLinear(hidden_dim, 2 * output_dim)
        else:
            self.output_layer = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)

        x_ = self.input_layer(x)
        x_ = F.relu(x_)
        for hidden_layer in self.hidden_layers:
            x_ = hidden_layer(x_)
            x_ = F.relu(x_)
        x_ = self.output_layer(x_)
        # x_ = F.relu(x_)
        return x_


    def get_loss(self, inputs, labels,  sample_nbr, complexity_cost_weight=1):
        labels = labels.view(labels.shape[0], -1)
        likelihood_cost = 0
        complexity_cost = 0
        # Array to collect the outputs
        for _ in range(sample_nbr):
            outputs = self(inputs)
            if self.hetero_noise_est:
                likelihood_cost += self.criterion(outputs[..., :self.output_dim], labels, outputs[..., self.output_dim:].exp())
            else:
                likelihood_cost += self.criterion(outputs, labels)
            complexity_cost += self.nn_kl_divergence() * complexity_cost_weight

        loss = (likelihood_cost + complexity_cost) / sample_nbr
        return loss

    def sample_detailed_loss(self, inputs, labels, sample_nbr, complexity_cost_weight=1):
        labels = labels.view(labels.shape[0], -1)
        likelihood_cost = 0
        complexity_cost = 0
        # Array to collect the outputs
        means, noise_stds = [], []
        for _ in range(sample_nbr):
            outputs = self(inputs)
            # y_hat.append(outputs.cpu().detach().numpy())
            if self.hetero_noise_est:
                means.append(outputs[:, :self.output_dim][..., None])
                noise_stds.append(outputs[:, self.output_dim:].exp()[..., None])
                likelihood_cost += self.criterion(outputs[..., :self.output_dim], labels, outputs[..., self.output_dim:].exp())
            else:
                means.append(outputs[:, :][..., None])
                likelihood_cost += self.criterion(outputs, labels)
            complexity_cost += self.nn_kl_divergence() * complexity_cost_weight

        # y_hat = np.array(y_hat)
        if self.hetero_noise_est:
            means, noise_stds = torch.cat(means, dim=-1), torch.cat(noise_stds, dim=-1)
            mean = means.mean(dim=-1)
            alea = noise_stds.mean(dim=-1)
            std = (means.std(dim=-1) ** 2 + alea ** 2) ** 0.5
        else:
            means = torch.cat(means, dim=-1)
            mean = means.mean(dim=-1)
            std = means.std(dim=-1)
            alea = torch.zeros_like(std).to(get_device())

        mse = ((mean - labels) ** 2).mean()
        loss = (likelihood_cost + complexity_cost) / sample_nbr
        return np.array(mean.detach().cpu()), \
               np.array(std.detach().cpu()), \
               np.array(alea.detach().cpu()), \
               loss.detach().cpu(), mse.detach().cpu()

    def get_accuracy_matrix(self, inputs, labels, sample_nbr):
        labels = labels.view(labels.shape[0], -1)
        means, noise_stds = [], []
        for _ in range(sample_nbr):
            outputs = self(inputs)
            means.append(outputs[:, :self.output_dim][..., None])

        means = torch.cat(means, dim=-1)
        mean = means.mean(dim=-1)

        mse = ((mean - labels) ** 2).mean()
        mape = ((mean - labels) / (labels+1e-10)).abs().mean()

        return mse.detach().cpu(), mape.detach().cpu()

    def save(self, dir):
        checkpoint = {'model': self.state_dict(), 'norm_mean': self.norm_mean, 'norm_std': self.norm_std}
        torch.save(checkpoint, dir)

    def load(self, dir):
        checkpoint = torch.load(dir + '/model.pt')
        self.load_state_dict(checkpoint['model'])
        self.norm_mean = checkpoint['norm_mean']
        self.norm_std = checkpoint['norm_std']


# def evaluate_regression(regressor, X, y, samples=100, std_multiplier=2):
#     preds = [regressor(X) for i in range(samples)]
#     preds = torch.stack(preds)
#     means = preds.mean(axis=0)
#     stds = preds.std(axis=0)
#     ci_upper = means + (std_multiplier * stds)
#     ci_lower = means - (std_multiplier * stds)
#     ic_acc = (ci_lower <= y) * (ci_upper >= y)
#     ic_acc = ic_acc.float().mean()
#     return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()

@variational_estimator
class BayesianClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_depth, criterion):
        super().__init__()
        # self.hetero_noise_est = hetero_noise_est
        self.criterion = criterion
        self.input_dim = input_dim
        self.num_classes = output_dim
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


    def get_loss(self, inputs, labels, sample_nbr, complexity_cost_weight=1):
        likelihood_cost = 0
        complexity_cost = 0
        for _i in range(sample_nbr):
            outputs = self(inputs)
            y = one_hot_embedding(labels, self.num_classes)
            _, preds = torch.max(outputs, 1)

            likelihood_cost += self.criterion(outputs, y.float(), reduction='sum')

            complexity_cost += self.nn_kl_divergence() * complexity_cost_weight

        loss = (likelihood_cost + complexity_cost) / sample_nbr
        return loss

    def sample_detailed_loss(self, inputs, labels, sample_nbr, complexity_cost_weight=1):
        likelihood_cost = 0
        complexity_cost = 0

        means = []
        for _i in range(sample_nbr):
            outputs = self(inputs)
            y = one_hot_embedding(labels, self.num_classes)
            _, preds = torch.max(outputs, 1)

            means.append(outputs[..., None])
            likelihood_cost += self.criterion(outputs, y.float(), reduction='sum')
            complexity_cost += self.nn_kl_divergence() * complexity_cost_weight

        means = torch.cat(means, dim=-1)
        probs = F.softmax(means, dim=1)
        prob = probs.mean(dim=-1)
        mean = means.mean(dim=-1)
        _, preds = torch.max(mean, 1)
        # prob = F.softmax(mean, dim=1)
        loss = (likelihood_cost + complexity_cost) / sample_nbr

        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
        acc = torch.mean(match)

        return np.array(mean.detach().cpu()), \
            np.array(prob.detach().cpu()), \
            loss.detach().cpu(), acc.detach().cpu()

@variational_estimator
class BayesianLeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.criterion = F.cross_entropy
        self.conv1 = BayesianConv2d(3, 16, (5, 5))
        self.conv2 = BayesianConv2d(16, 32, (5, 5))
        self.fc1 = BayesianLinear(5*5*32, 120)
        self.fc2 = BayesianLinear(120, 84)
        self.fc3 = BayesianLinear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def get_loss(self, inputs, labels, sample_nbr, complexity_cost_weight=1):
        likelihood_cost = 0
        complexity_cost = 0
        for _i in range(sample_nbr):
            outputs = self(inputs)
            y = one_hot_embedding(labels, self.num_classes)
            _, preds = torch.max(outputs, 1)

            likelihood_cost += self.criterion(outputs, y.float(), reduction='sum')

            complexity_cost += self.nn_kl_divergence() * complexity_cost_weight

        loss = (likelihood_cost + complexity_cost) / sample_nbr
        return loss

    def sample_detailed_loss(self, inputs, labels, sample_nbr, complexity_cost_weight=1):
        likelihood_cost = 0
        complexity_cost = 0

        means = []
        for _i in range(sample_nbr):
            outputs = self(inputs)
            y = one_hot_embedding(labels, self.num_classes)
            _, preds = torch.max(outputs, 1)

            means.append(outputs[..., None])
            likelihood_cost += self.criterion(outputs, y.float(), reduction='sum')
            complexity_cost += self.nn_kl_divergence() * complexity_cost_weight

        means = torch.cat(means, dim=-1)
        probs = F.softmax(means, dim=1)
        prob = probs.mean(dim=-1)
        mean = means.mean(dim=-1)
        _, preds = torch.max(mean, 1)
        # prob = F.softmax(mean, dim=1)
        loss = (likelihood_cost + complexity_cost) / sample_nbr

        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
        acc = torch.mean(match)

        return np.array(mean.detach().cpu()), \
            np.array(prob.detach().cpu()), \
            loss.detach().cpu(), acc.detach().cpu()