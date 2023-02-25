import torch
from torch.nn import Linear
import torch.nn.functional as F
from edl.layers.dense import DenseNormalGamma
from edl.loss.coutinues import RegularizedGammaLoss
from edl.loss.discrete import relu_evidence
import numpy as np
from lib.utils import get_device, one_hot_embedding


class EvidentialClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_depth, output_dim, annealing_step, criterion):
        super(EvidentialClassifier, self).__init__()
        self.annealing_step = annealing_step
        self.criterion = criterion
        self.num_classes = output_dim
        self.input_layer = Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList(
            [Linear(hidden_dim, hidden_dim) for i in range(hidden_depth)]
        )
        self.output_layer = Linear(hidden_dim, output_dim)
        # self.loss_func = RegularizedGammaLoss


    def forward(self, x):
        x_ = self.input_layer(x)
        x_ = F.relu(x_)
        for hidden_layer in self.hidden_layers:
            x_ = hidden_layer(x_)
            x_ = F.relu(x_)
        x_ = self.output_layer(x_)
        return x_


    def get_loss(self, inputs, labels, epoch, complexity_cost_weight=1):
        outputs = self(inputs)
        y = one_hot_embedding(labels, self.num_classes)
        _, preds = torch.max(outputs, 1)
        loss = self.criterion(outputs, y.float(), epoch, self.num_classes, self.annealing_step, complexity_cost_weight)
        return loss

    def sample_detailed_loss(self, inputs, labels, epoch, complexity_cost_weight=1):
        outputs = self(inputs)
        y = one_hot_embedding(labels, self.num_classes)
        # device = get_device()
        # y = y.to(device)
        _, preds = torch.max(outputs, 1)
        loss = self.criterion(outputs, y.float(), epoch, self.num_classes, self.annealing_step, complexity_cost_weight)

        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
        acc = torch.mean(match)
        evidence = relu_evidence(outputs)
        alpha = evidence + 1
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        # prob, _ = torch.max(alpha / torch.sum(alpha, dim=1, keepdim=True), 1)

        # u = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)
        # total_evidence = torch.sum(evidence, 1, keepdim=True)
        # mean_evidence = torch.mean(total_evidence)
        # mean_evidence_succ = torch.sum(
        #     torch.sum(evidence, 1, keepdim=True) * match
        # ) / torch.sum(match + 1e-20)
        # mean_evidence_fail = torch.sum(
        #     torch.sum(evidence, 1, keepdim=True) * (1 - match)
        # ) / (torch.sum(torch.abs(1 - match)) + 1e-20)

        return np.array(preds.detach().cpu()), \
               np.array(prob.detach().cpu()), \
               loss.detach().cpu(), acc.detach().cpu()


class EvidentialRegressor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_depth, output_dim=1):
        super(EvidentialRegressor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList(
            [Linear(hidden_dim, hidden_dim) for i in range(hidden_depth)]
        )
        self.output_layer = DenseNormalGamma(hidden_dim, output_dim)
        self.loss_func = RegularizedGammaLoss


    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x_ = self.input_layer(x)
        x_ = F.relu(x_)
        for hidden_layer in self.hidden_layers:
            x_ = hidden_layer(x_)
            x_ = F.relu(x_)
        x_ = self.output_layer(x_)
        return x_

    def sample_detailed_loss(self, inputs, labels):
        outputs = self(inputs)
        gamma, nu, alpha, beta = outputs
        mean = gamma
        aleatoric = beta/(alpha-1)
        epistemic = beta/nu/(alpha-1)
        std = (aleatoric + epistemic) ** 0.5
        alea = aleatoric ** 0.5

        labels = labels.view(labels.shape[0], -1)
        loss = self.loss_func(outputs, labels)
        mse = ((mean - labels) ** 2).mean()

        return np.array(mean.detach().cpu()), \
               np.array(std.detach().cpu()), \
               np.array(alea.detach().cpu()), \
               loss.detach().cpu(), mse.detach().cpu()

    def get_accuracy_matrix(self, inputs, labels):
        labels = labels.view(labels.shape[0], -1)
        means, noise_stds = [], []

        for _ in range(1):
            outputs, _, _, _ = self(inputs)
            means.append(outputs[..., None])

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