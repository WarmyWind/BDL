import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from lib.utils import get_device, one_hot_embedding

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.loss_func = F.cross_entropy
        self.conv1 = nn.Conv2d(3, 16, (5, 5))
        self.conv2 = nn.Conv2d(16, 32, (5, 5))
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)


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


    def get_loss(self, inputs, labels):
        outputs = self(inputs)
        y = one_hot_embedding(labels, self.num_classes)
        _, preds = torch.max(outputs, 1)
        loss = self.loss_func(outputs, y.float(), reduction='mean')

        return loss


    def sample_detailed_loss(self, inputs, labels, sample_nbr=1):
        means = []
        for _i in range(sample_nbr):
            outputs = self(inputs)
            means.append(outputs[..., None])

        means = torch.cat(means, dim=-1)
        mean = means.mean(dim=-1)

        _, preds = torch.max(mean, 1)
        probs = F.softmax(mean, dim=1)
        y = one_hot_embedding(labels, self.num_classes)
        loss = self.loss_func(mean, y.float(), reduction='mean')

        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
        acc = torch.mean(match)

        return np.array(mean.detach().cpu()), \
            np.array(probs.detach().cpu()), \
            loss.detach().cpu(), acc.detach().cpu()

