# %%

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Average, Loss
from ignite.contrib.handlers import ProgressBar

import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood

from due.dkl import DKL, GP, initial_values
from due.sngp import Laplace
from due.fc_resnet import FCResNet

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_palette("colorblind")


# %%

def make_data(n_samples, noise=0.05, seed=2):
    # make some random sines & cosines
    np.random.seed(seed)
    n_samples = int(n_samples)

    W = np.random.randn(30, 1)
    b = np.random.rand(30, 1) * 2 * np.pi

    x = 5 * np.sign(np.random.randn(n_samples)) + np.random.randn(n_samples).clip(-2, 2)
    y = np.cos(W * x + b).sum(0) + noise * np.random.randn(n_samples)
    return x[..., None], y


# %%

n_samples = 1e3
# n_samples = 1e6

domain = 15

x, y = make_data(n_samples)
plt.scatter(x, y)

# %%

np.random.seed(0)
torch.manual_seed(0)

batch_size = 128

X_train, y_train = make_data(n_samples)
X_test, y_test = X_train, y_train

ds_train = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)

ds_test = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
dl_test = torch.utils.data.DataLoader(ds_test, batch_size=512, shuffle=False)

steps = 5e3
epochs = steps // len(dl_train) + 1
print(f"Training with {n_samples} datapoints for {epochs} epochs")

# Change this boolean to False for SNGP
DUE = True

input_dim = 1
features = 128
depth = 4
num_outputs = 1  # regression with 1D output
spectral_normalization = True
coeff = 0.95
n_power_iterations = 1
dropout_rate = 0.01

feature_extractor = FCResNet(
    input_dim=input_dim,
    features=features,
    depth=depth,
    spectral_normalization=spectral_normalization,
    coeff=coeff,
    n_power_iterations=n_power_iterations,
    dropout_rate=dropout_rate
)

if DUE:
    n_inducing_points = 20
    kernel = "RBF"

    initial_inducing_points, initial_lengthscale = initial_values(
        ds_train, feature_extractor, n_inducing_points
    )

    gp = GP(
        num_outputs=num_outputs,
        initial_lengthscale=initial_lengthscale,
        initial_inducing_points=initial_inducing_points,
        kernel=kernel,
    )

    model = DKL(feature_extractor, gp)

    likelihood = GaussianLikelihood()
    elbo_fn = VariationalELBO(likelihood, model.gp, num_data=len(ds_train))
    loss_fn = lambda x, y: -elbo_fn(x, y)
else:
    num_gp_features = 128
    num_random_features = 1024
    normalize_gp_features = True
    feature_scale = 2
    ridge_penalty = 1

    model = Laplace(feature_extractor,
                    features,
                    num_gp_features,
                    normalize_gp_features,
                    num_random_features,
                    num_outputs,
                    len(ds_train),
                    batch_size,
                    ridge_penalty=ridge_penalty,
                    feature_scale=feature_scale
                    )

    loss_fn = F.mse_loss

if torch.cuda.is_available():
    model = model.cuda()
    if DUE:
        likelihood = likelihood.cuda()

lr = 1e-3

parameters = [
    {"params": model.parameters(), "lr": lr},
]

if DUE:
    parameters.append({"params": likelihood.parameters(), "lr": lr})

optimizer = torch.optim.Adam(parameters)
pbar = ProgressBar()


def step(engine, batch):
    model.train()
    if DUE:
        likelihood.train()

    optimizer.zero_grad()

    x, y = batch
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    y_pred = model(x)

    if not DUE:
        y_pred.squeeze_()

    loss = loss_fn(y_pred, y)

    loss.backward()
    optimizer.step()

    return loss.item()


def eval_step(engine, batch):
    model.eval()
    if DUE:
        likelihood.eval()

    x, y = batch
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()

    y_pred = model(x)

    return y_pred, y


trainer = Engine(step)
evaluator = Engine(eval_step)

metric = Average()
metric.attach(trainer, "loss")
pbar.attach(trainer)

if DUE:
    metric = Loss(lambda y_pred, y: - likelihood.expected_log_prob(y, y_pred).mean())
else:
    metric = Loss(lambda y_pred, y: F.mse_loss(y_pred[0].squeeze(), y))

metric.attach(evaluator, "loss")


@trainer.on(Events.EPOCH_COMPLETED(every=int(epochs / 10) + 1))
def log_results(trainer):
    evaluator.run(dl_test)
    print(f"Results - Epoch: {trainer.state.epoch} - "
          f"Test Likelihood: {evaluator.state.metrics['loss']:.2f} - "
          f"Loss: {trainer.state.metrics['loss']:.2f}")


if not DUE:
    @trainer.on(Events.EPOCH_STARTED)
    def reset_precision_matrix(trainer):
        model.reset_precision_matrix()

# %%

trainer.run(dl_train, max_epochs=epochs)

# %%

model.eval()
if DUE:
    likelihood.eval()

x_lin = np.linspace(-domain, domain, 100)

with torch.no_grad(), gpytorch.settings.num_likelihood_samples(64):
    xx = torch.tensor(x_lin[..., None]).float()
    if torch.cuda.is_available():
        xx = xx.cuda()
    pred = model(xx)

    if DUE:
        ol = likelihood(pred)
        output = ol.mean.cpu()
        output_std = ol.stddev.cpu()
    else:
        output = pred[0].squeeze().cpu()
        output_var = pred[1].diagonal()
        output_std = output_var.sqrt().cpu()

# %%

plt.xlim(-domain, domain)
plt.ylim(-10, 10)
plt.fill_between(x_lin, output - output_std, output + output_std, alpha=0.2, color='b')
plt.fill_between(x_lin, output - 2 * output_std, output + 2 * output_std, alpha=0.2, color='b')

plt.scatter([], [])
plt.scatter([], [])
X_vis, y_vis = make_data(n_samples=300)

plt.scatter(X_vis.squeeze(), y_vis, facecolors='none', edgecolors='g', linewidth=2)
plt.plot(x_lin, output, alpha=0.5)

# %%

plt.xlim(-domain, domain)

for i in range(12):
    plt.plot(x_lin, ol.rsample().cpu(), alpha=0.3, color='b')

plt.scatter([], [])
plt.scatter([], [])
X_vis, y_vis = make_data(n_samples=200)
plt.scatter(X_vis.squeeze(), y_vis, s=50)

# %%

# Inspired by https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
# only works on 1,000 samples

train_x, train_y = torch.tensor(x).float(), torch.tensor(y).float()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = GaussianLikelihood()
exact_gp = ExactGPModel(train_x, train_y, likelihood)

exact_gp.train()
likelihood.train()

optimizer = torch.optim.Adam(exact_gp.parameters(), lr=0.1)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, exact_gp)

training_iter = 100

for i in range(training_iter):
    optimizer.zero_grad()
    output = exact_gp(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        exact_gp.covar_module.base_kernel.lengthscale.item(),
        exact_gp.likelihood.noise.item()
    ))
    optimizer.step()

# %%

exact_gp.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(-domain, domain, 100)
    observed_pred = likelihood(exact_gp(test_x))

    output = observed_pred.mean
    output_std = observed_pred.stddev

plt.xlim(-domain, domain)
plt.fill_between(x_lin, output - output_std, output + output_std, alpha=0.2, color='b')
plt.fill_between(x_lin, output - 2 * output_std, output + 2 * output_std, alpha=0.2, color='b')

plt.scatter([], [])
plt.scatter([], [])
X_vis, y_vis = make_data(n_samples=300)

plt.scatter(X_vis.squeeze(), y_vis, facecolors='none', edgecolors='g', linewidth=2)
plt.plot(x_lin, output, alpha=0.5)