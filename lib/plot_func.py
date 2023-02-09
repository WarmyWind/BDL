import numpy as np
import matplotlib
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from typing import Union, Tuple, List, Any, NoReturn
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from lib.utils import one_hot_embedding
from scipy import stats

def calc_bins(probs, labels, num_bins, marginal):
    labels_oneh = one_hot_embedding(labels, probs.shape[-1])
    # Assign each prediction to a bin
    bins = np.linspace(1/num_bins, 1, num_bins)
    if torch.is_tensor(probs):
        probs = np.array(probs.detach().cpu())
    top_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    binned = np.digitize(probs, bins, right=True)
    top_binned = np.digitize(top_probs, bins, right=True)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    if probs.shape[1] == 2 or marginal == True:  # binary classification
        for bin in range(num_bins):
            bin_sizes[bin] = len(probs[binned == bin])
            if bin_sizes[bin] > 0:
                bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
                bin_confs[bin] = (probs[binned == bin]).sum() / bin_sizes[bin]
    else:  # multi-classification
        for bin in range(num_bins):
            bin_sizes[bin] = len(top_probs[top_binned == bin])
            if bin_sizes[bin] > 0:
                bin_labels = labels.detach().cpu().numpy()[top_binned == bin]
                bin_preds = preds[top_binned == bin]
                bin_accs[bin] = np.sum(bin_labels == bin_preds) / bin_sizes[bin]
                bin_confs[bin] = np.sum(top_probs[top_binned == bin]) / bin_sizes[bin]
                # bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
                # bin_confs[bin] = (probs[binned == bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds, labels, num_bins, marginal):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels, num_bins, marginal)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE

def draw_reliability_graph(probs, labels, num_bins=10, recal_model=None, marginal=True):
    if recal_model is not None:
        _shape = probs.shape
        probs = recal_model.predict(probs.flatten())
        probs = np.reshape(probs, _shape)

    ECE, MCE = get_metrics(probs, labels, num_bins, marginal)
    bins, _, bin_accs, _, bin_sizes = calc_bins(probs, labels, num_bins, marginal)

    # fig = plt.figure(figsize=(8, 8))
    fig, ax = plt.subplots(figsize=(8, 10))
    fig.clf()

    from matplotlib import gridspec
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])

    # x/y limits
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0, 1.0)

    # x/y labels
    # ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')

    # Create grid
    # ax.set_axisbelow(True)
    ax1.grid(color='gray', linestyle='dashed')

    # Error bars
    # plt.bar(bins-1/2/num_bins, bins,  width=1/num_bins, alpha=0.3, edgecolor='black', color='r', hatch='\\')
    ax1.bar(bins - 1 / 2 / num_bins, bins, width=1 / num_bins, alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Draw bars and identity line
    # plt.bar(bins-1/2/num_bins, bin_accs, width=1/num_bins, alpha=1, edgecolor='black', color='b')
    # plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)
    ax1.bar(bins - 1 / 2 / num_bins, bin_accs, width=1 / num_bins, alpha=1, edgecolor='black', color='b')
    ax1.plot([0,1],[0,1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    # plt.gca().set_aspect('equal', adjustable='box')
    # ax.set_aspect('equal', adjustable='box')

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
    # MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
    ax1.legend(handles=[ECE_patch])

    ax2.set_xlim(0, 1.0)
    # ax2.set_ylim(0, 5)
    ax2.bar(bins - 1 / 2 / num_bins, bin_sizes, width=1 / num_bins, alpha=1, edgecolor='black', color='b')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    return fig, ax

def plot_calibration(
    exp_props: Union[np.ndarray, None] = None,
    obs_props: Union[np.ndarray, None] = None,
    recal_model: Any = None,
    ax: Union[matplotlib.axes.Axes, None] = None,
):
    if recal_model is not None:
        obs_props = recal_model.predict(obs_props)
    else:
        obs_props = obs_props

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", label="Ideal", c="#ff7f0e")
    ax.plot(exp_props, obs_props, label='Predictor', c="#1f77b4")
    ax.fill_between(exp_props, exp_props, obs_props, alpha=0.2)

    # Format plot
    ax.set_xlabel("Predicted Proportion")
    ax.set_ylabel("Observed Proportion")
    ax.axis("square")

    buff = 0.01
    ax.set_xlim([0 - buff, 1 + buff])
    ax.set_ylim([0 - buff, 1 + buff])

    ax.set_title("Average Calibration")

    # Compute miscalibration area
    polygon_points = []
    for point in zip(exp_props, obs_props):
        polygon_points.append(point)
    for point in zip(reversed(exp_props), reversed(exp_props)):
        polygon_points.append(point)
    polygon_points.append((exp_props[0], obs_props[0]))
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy  # original data
    ls = LineString(np.c_[x, y])  # closed, non-simple
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list = [poly.area for poly in polygonize(mls)]
    miscalibration_area = np.asarray(polygon_area_list).sum()

    # Annotate plot with the miscalibration area
    ax.text(
        x=0.95,
        y=0.05,
        s="Miscalibration area = %.2f" % miscalibration_area,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize="small",
    )
    return ax

def plot_uncertainty(x, y, y_hat, std):
    x, y, y_hat, std = x.squeeze(), y.squeeze(), y_hat.squeeze(), std.squeeze()

    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    # Get upper and lower confidence bounds
    lower, upper = y_hat - std*3, y_hat + std*3
    # Plot training data as black stars

    ax.plot(np.arange(len(x)+len(y)), np.concatenate([x, y]), 'k*-')
    # Plot predictive means as blue line
    ax.plot(np.arange(len(x), len(x)+len(y)), y_hat, 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(np.arange(len(x), len(x)+len(y)), lower, upper, alpha=0.5)
    # ax.set_ylim([-3, 3])
    ax.legend(['True Data', 'Prediction', 'Uncertainty'])

    return f, ax


def plot_bar(data, xlabel, ylabel, para_list, label_list, ax=None):
    if ax == None:
        fig, ax = plt.subplots()

    # 创建分组柱状图，需要自己控制x轴坐标
    xticks = np.arange(len(para_list))

    # 注意控制柱子的宽度，这里选择0.25
    for i in range(len(label_list)):
        ax.bar(xticks + i * 0.25, data[i], width=0.25, label=label_list[i])

    if xlabel != '':
        ax.set_xlabel(xlabel)
    if ylabel != '':
        ax.set_ylabel(ylabel)
    ax.legend()

    # 最后调整x轴标签的位置
    ax.set_xticks(xticks + 0.25*(len(label_list)-1)/2)
    ax.set_xticklabels(para_list)

    return ax



def plot_sample_uncertainty(x_test, y_pred, y_std, y_true, test_idx, in_exp_proportions=0.95, prop_type="interval",
                            recal_model=None, ax=None):
    residuals = y_pred - y_true
    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(-1, 1)
    norm = stats.norm(loc=0, scale=1)
    if prop_type == "interval":
        gaussian_lower_bound = norm.ppf(0.5 - in_exp_proportions / 2.0)
        gaussian_upper_bound = norm.ppf(0.5 + in_exp_proportions / 2.0)
        test_upper_bound = gaussian_upper_bound * y_std[test_idx] + y_pred[test_idx]
        test_lower_bound = gaussian_lower_bound * y_std[test_idx] + y_pred[test_idx]

        above_lower = normalized_residuals >= gaussian_lower_bound
        below_upper = normalized_residuals <= gaussian_upper_bound
        within_quantile = above_lower * below_upper
        obs_proportions = np.sum(within_quantile, axis=0).flatten()[0] / len(residuals)
        obs_gaussian_lower_bound = norm.ppf(0.5 - obs_proportions / 2.0)
        obs_gaussian_upper_bound = norm.ppf(0.5 + obs_proportions / 2.0)

        true_test_std = y_std[test_idx] * gaussian_upper_bound / obs_gaussian_upper_bound
        true_upper_bound = gaussian_upper_bound * true_test_std + y_pred[test_idx]
        true_lower_bound = gaussian_lower_bound * true_test_std + y_pred[test_idx]

        if recal_model != None:
            calibrated_exp_proportions = recal_model.predict(np.array([in_exp_proportions]))
            calibrated_exp_proportions = calibrated_exp_proportions[0]
        else:
            calibrated_exp_proportions = in_exp_proportions
        gaussian_lower_bound = norm.ppf(0.5 - calibrated_exp_proportions / 2.0)
        gaussian_upper_bound = norm.ppf(0.5 + calibrated_exp_proportions / 2.0)
        test_upper_bound = gaussian_upper_bound * y_std[test_idx] + y_pred[test_idx]
        test_lower_bound = gaussian_lower_bound * y_std[test_idx] + y_pred[test_idx]

    else:
        # gaussian_quantile_bound = norm.ppf(in_exp_proportions)
        # below_quantile = normalized_residuals <= gaussian_quantile_bound
        # obs_proportions = np.sum(below_quantile, axis=0).flatten() / len(residuals)
        raise Exception("Unsupported prop_type!")
    if ax == None:
        f, ax = plt.subplots(1, 1)


    y_test = y_true[test_idx][..., None]
    y_hat = y_pred[test_idx][..., None]
    # Plot training data as black stars
    ax.plot(np.arange(len(x_test) + len(y_test)), np.concatenate([x_test, y_test]), 'k*-')
    # Plot predictive means as red line
    ax.plot(np.arange(len(x_test) + len(y_test)), np.concatenate([x_test, y_hat]), 'r')


    # Shade between the lower and upper confidence bounds
    test_lower_bound = [x_test[-1], test_lower_bound]
    test_upper_bound = [x_test[-1], test_upper_bound]
    true_lower_bound = [x_test[-1], true_lower_bound]
    true_upper_bound = [x_test[-1], true_upper_bound]
    if test_lower_bound[-1] > true_lower_bound[-1]:
        ax.fill_between(np.arange(len(x_test) - 1, len(x_test) + len(y_test)), true_lower_bound, test_lower_bound,
                        alpha=0.4, color='blue')
        ax.fill_between(np.arange(len(x_test) - 1, len(x_test) + len(y_test)), test_lower_bound, test_upper_bound,
                        alpha=0.4, color='yellow')
        ax.fill_between(np.arange(len(x_test) - 1, len(x_test) + len(y_test)), test_upper_bound, true_upper_bound,
                        alpha=0.4, color='blue')
        ax.legend(['True Data', 'Prediction', 'True {} Interval'.format(in_exp_proportions), 'Predictive {} Interval'.format(in_exp_proportions)])
    else:
        ax.fill_between(np.arange(len(x_test) - 1, len(x_test) + len(y_test)), test_lower_bound, true_lower_bound,
                        alpha=0.4, color='yellow')
        ax.fill_between(np.arange(len(x_test) - 1, len(x_test) + len(y_test)), true_lower_bound, true_upper_bound,
                        alpha=0.4, color='blue')
        ax.fill_between(np.arange(len(x_test) - 1, len(x_test) + len(y_test)), true_upper_bound, test_upper_bound,
                        alpha=0.4, color='yellow')
        ax.legend(['True Data', 'Prediction', 'Predictive {} Interval'.format(in_exp_proportions), 'True {} Interval'.format(in_exp_proportions)])

    ax.set_xlabel("Time")
    # ax.set_ylabel("Value")
    return ax
