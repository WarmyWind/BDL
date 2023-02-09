import torch
from torch import nn, optim
from torch.nn import functional as F
from lib.utils import one_hot_embedding
import numpy as np
from lib.plot_func import get_metrics

class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.3)

    def forward(self, input, sample_nbr=10):
        _temp_logits = []
        for ii in range(sample_nbr):
            outputs = self.model(input)
            _temp_logits.append(outputs[..., None])
        _temp_logits = torch.cat(_temp_logits, dim=-1)
        logits = _temp_logits.mean(dim=-1)
        # logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, sample_nbr=10):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        # ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                _temp_logits = []
                for ii in range(sample_nbr):
                    outputs = self.model(input)
                    _temp_logits.append(outputs[..., None])
                _temp_logits = torch.cat(_temp_logits, dim=-1)
                logits = _temp_logits.mean(dim=-1)
                # logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda().long()
            # if labels.shape != logits.shape:
            #     labels = one_hot_embedding(labels, logits.shape[-1])

        # # Calculate NLL and ECE before temperature scaling
        # before_temperature_nll = nll_criterion(logits, labels).item()
        # # before_temperature_ece = ece_criterion(logits, labels).item()
        # before_temperature_ece, before_temperature_mce = get_metrics(F.softmax(logits, dim=1), labels, num_bins=15, marginal=True)
        # print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece*100))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=10000, line_search_fn='strong_wolfe')

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)

            loss.backward()
            return loss
        optimizer.step(eval)

        # # Calculate NLL and ECE after temperature scaling
        # after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        # # after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        # after_temperature_ece, after_temperature_mce = get_metrics(F.softmax(self.temperature_scale(logits), dim=1),
        #                                                            labels, num_bins=15, marginal=True)
        # print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece*100))

        print('Optimal temperature: %.3f' % self.temperature.item())
        return self

    def sample_detailed_loss(self, inputs, labels, sample_nbr=1):
        means = []
        for _i in range(sample_nbr):
            outputs = self(inputs)
            # y = one_hot_embedding(labels, self.num_classes)
            # _, preds = torch.max(outputs, 1)
            means.append(outputs[..., None])

        means = torch.cat(means, dim=-1)
        mean = means.mean(dim=-1)

        _, preds = torch.max(mean, 1)
        probs = F.softmax(mean, dim=1)
        y = one_hot_embedding(labels, self.model.num_classes)
        # loss = self.model.loss_func(mean, y.float(), reduction='sum')

        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
        acc = torch.mean(match)

        return np.array(mean.detach().cpu()), \
            np.array(probs.detach().cpu()), \
            acc.detach().cpu()


# class _ECELoss(nn.Module):
#     """
#     Calculates the Expected Calibration Error of a model.
#     (This isn't necessary for temperature scaling, just a cool metric).
#     The input to this loss is the logits of a model, NOT the softmax scores.
#     This divides the confidence outputs into equally-sized interval bins.
#     In each bin, we compute the confidence gap:
#     bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
#     We then return a weighted average of the gaps, based on the number
#     of samples in each bin
#     See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
#     "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
#     2015.
#     """
#     def __init__(self, n_bins=15):
#         """
#         n_bins (int): number of confidence interval bins
#         """
#         super(_ECELoss, self).__init__()
#         bin_boundaries = torch.linspace(0, 1, n_bins + 1)
#         self.bin_lowers = bin_boundaries[:-1]
#         self.bin_uppers = bin_boundaries[1:]
#
#     def forward(self, logits, labels):
#         softmaxes = F.softmax(logits, dim=1)
#         confidences, predictions = torch.max(softmaxes, 1)
#         accuracies = predictions.eq(labels)
#
#         ece = torch.zeros(1, device=logits.device)
#         for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
#             # Calculated |confidence - accuracy| in each bin
#             in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
#             prop_in_bin = in_bin.float().mean()
#             if prop_in_bin.item() > 0:
#                 accuracy_in_bin = accuracies[in_bin].float().mean()
#                 avg_confidence_in_bin = confidences[in_bin].mean()
#                 ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
#
#         return ece