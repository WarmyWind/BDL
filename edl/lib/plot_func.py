import numpy as np
import matplotlib
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from typing import Union, Tuple, List, Any, NoReturn
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union
from lib.utils import one_hot_embedding

def calc_bins(probs, labels, num_bins=15):
    labels_oneh = one_hot_embedding(labels, probs.shape[-1])
    # Assign each prediction to a bin
    bins = np.linspace(1/num_bins, 1, num_bins)
    if torch.is_tensor(probs):
        probs = np.array(probs.detach().cpu())

    binned = np.digitize(probs, bins, right=True)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(probs[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (probs[binned == bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds, labels, num_bins):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels, num_bins)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)

    return ECE, MCE

def draw_reliability_graph(probs, labels, num_bins=10, recal_model=None):
    if recal_model is not None:
        _shape = probs.shape
        probs = recal_model.predict(probs.flatten())
        probs = np.reshape(probs, _shape)

    ECE, MCE = get_metrics(probs, labels, num_bins)
    bins, _, bin_accs, _, _ = calc_bins(probs, labels, num_bins)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(bins-1/2/num_bins, bins,  width=1/num_bins, alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Draw bars and identity line
    plt.bar(bins-1/2/num_bins, bin_accs, width=1/num_bins, alpha=1, edgecolor='black', color='b')
    plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
    # MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
    plt.legend(handles=[ECE_patch])

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