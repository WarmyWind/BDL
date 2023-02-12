import torch
from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
# import torch_geometric
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
from dnn.model import DNNClassifier, DNNBinaryClassifier
from bnn.model import BayesianClassifier
from mcdropout.model import MCDropoutClassifier
from ensemble.model import BootstrapEnsemble
from edl.model import EvidentialClassifier
from edl.loss.discrete import edl_digamma_loss, edl_log_loss, edl_mse_loss
from Dataset.load_dataset import *
from utils import build_model
from lib.plot_func import *
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import uncertainty_toolbox as uct
import calibration as cal
from temperature_scaling import ModelWithTemperature

# Set plot style
uct.viz.set_style()
uct.viz.update_rc("text.usetex", True)  # Set to True for system latex
uct.viz.update_rc("font.size", 14)  # Set font size
uct.viz.update_rc("xtick.labelsize", 14)  # Set font size for xaxis tick labels
uct.viz.update_rc("ytick.labelsize", 14)  # Set font size for yaxis tick labels

def main(hparams):
    print("Use CUDA: ", torch.cuda.is_available())
    if hparams.task == 'HO_predict':
        dl_train, dl_calibration, dl_test = load_handover(hparams.dataset_path, hparams.input_dim, hparams.output_dim,
                                                    test_size=0.2, valid_size=0.2,
                                                    train_batch_size=hparams.batch_size,
                                                    valid_batch_size=hparams.batch_size, seed=hparams.seed)

    elif hparams.task == 'cifar10':
        dl_train, dl_calibration, dl_test = load_cifar10(valid_size=0.1, train_batch_size=hparams.batch_size,
                                                   seed=hparams.seed, cali_dataset_num=hparams.cali_dataset_num)
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        raise Exception('Invalid task!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method = hparams.method  # 'BNN' , 'MCDropout'
    model = build_model(hparams)
    if method != "Ensemble":
        state_dict = torch.load(hparams.output_dir + '/model.pt')
        model.load_state_dict(state_dict)
        model.to(device)
    else:
        model.load(hparams.output_dir)
        model.to(device)

    def test_step(batch, model):
        if method != 'MCDropout':
            model.eval()

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        if method == 'MCDropout':
            y_pred, y_prob, _, acc = model.sample_detailed_loss(inputs=x,
                                                      labels=y,
                                                      sample_nbr=hparams.sample_nbr)

        elif method == 'BNN':
            complexity_cost_weight = 1 / len(dl_train)
            y_pred, y_prob, _, acc = model.sample_detailed_loss(inputs=x,
                                                      labels=y,
                                                      sample_nbr=hparams.sample_nbr,
                                                      complexity_cost_weight=complexity_cost_weight)

        elif method == 'Ensemble':
            y_pred, y_prob, _, acc = model.sample_detailed_loss(inputs=x, labels=y)
        elif method == 'EDL':
            y_pred, y_prob, _, acc = model.sample_detailed_loss(inputs=x, labels=y, epoch=hparams.epochs)
        elif method == 'DNN':
            y_pred, y_prob, _, acc = model.sample_detailed_loss(inputs=x, labels=y)
        elif method == 'CNN':
            y_pred, y_prob, _, acc = model.sample_detailed_loss(inputs=x, labels=y)
        else:
            raise Exception('Invalid method!')

        return y_pred, y_prob, acc

    ################################### Before Calibration ########################################
    test_sample_num, match = 0, 0
    test_labels = torch.tensor([])
    y_pred, y_prob = np.empty((0, hparams.output_dim)), np.empty((0, hparams.output_dim))
    for batch in dl_test:
        # x = batch[0].detach().cpu().numpy()
        test_labels = torch.cat((test_labels, batch[1]))
        _y_pred, _y_prob, _acc = test_step(batch, model)
        y_pred = np.concatenate((y_pred, _y_pred), 0)
        y_prob = np.concatenate((y_prob, _y_prob), 0)
        test_sample_num += len(batch[1])
        match += len(batch[1]) * _acc

    acc = match/test_sample_num
    print("Test Accuracy: {}".format(acc))

    y_pred = y_pred.squeeze()
    y_prob = y_prob.squeeze()

    # Before recalibration, test
    num_bins = hparams.num_bins
    marginal = hparams.marginal
    test_labels_arr = test_labels.detach().cpu().numpy().astype('int32')
    if hparams.calibrator == 'platt_scaling':
        test_pred_prob = y_prob[:, 1]
        # ECE_before = cal.get_ece(test_pred_prob, test_labels_arr)
    ECE_before, MCE_before = get_metrics(y_prob, test_labels, num_bins, marginal)

    if hparams.plot == True:
        fig, ax = draw_reliability_graph(y_prob, test_labels, num_bins=num_bins, marginal=marginal)
        ax.title.set_text("Before Calibration")
        plt.show()

    ############################# Calibration #############################
    if hparams.calibrator == 'isotonic_regression':
        cali_labels = torch.tensor([])
        y_cali_prob = np.empty((0, hparams.output_dim))
        for batch in dl_calibration:
            cali_labels = torch.cat((cali_labels, batch[1]))
            _y_cali_pred, _y_cali_prob, _ = test_step(batch, model)
            y_cali_prob = np.concatenate((y_cali_prob, _y_cali_prob), 0)

        # y_cali_pred = y_cali_pred.squeeze()
        y_cali_prob = y_cali_prob.squeeze()
        bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_cali_prob, cali_labels, num_bins, marginal)
        bin_accs = np.concatenate([[0], bin_accs, [1]])
        bin_confs = np.concatenate([[0], bin_confs, [1]])
        recal_model = uct.iso_recal(bin_accs, bin_confs)

        _shape = y_prob.shape
        probs = recal_model.predict(y_prob.flatten())
        probs = np.reshape(probs, _shape)
        ECE_after, MCE_after = get_metrics(probs, test_labels, num_bins, marginal)
        if hparams.plot == True:
            fig, ax = draw_reliability_graph(y_prob, test_labels, num_bins=num_bins, recal_model=recal_model,
                                             marginal=marginal)
            ax.title.set_text("Calibration:{}".format(hparams.calibrator))

    elif hparams.calibrator == 'temperature_scaling':
        scaled_model = ModelWithTemperature(model)
        scaled_model.set_temperature(dl_calibration, sample_nbr=hparams.sample_nbr)

        y_prob = np.empty((0, hparams.output_dim))
        for x, labels in dl_test:
            x, labels = x.to(device), labels.to(device)
            _y_pred, _y_prob, _acc = scaled_model.sample_detailed_loss(x, labels, hparams.sample_nbr)
            y_prob = np.concatenate((y_prob, _y_prob), 0)

        ECE_after, MCE_after = get_metrics(y_prob, test_labels, num_bins, marginal)
        if hparams.plot == True:
            fig, ax = draw_reliability_graph(y_prob, test_labels, num_bins=num_bins, marginal=marginal)
            ax.title.set_text("Calibration: {}".format(hparams.calibrator))

    elif hparams.calibrator == 'platt_scaling':
        cali_labels = torch.tensor([])
        y_cali_prob = np.empty((0, hparams.output_dim))
        for batch in dl_calibration:
            cali_labels = torch.cat((cali_labels, batch[1]))
            _y_cali_pred, _y_cali_prob, _ = test_step(batch, model)
            y_cali_prob = np.concatenate((y_cali_prob, _y_cali_prob), 0)

        num_points = len(y_cali_prob)
        cali_labels_arr = cali_labels.detach().cpu().numpy().astype('int32')

        zs = y_cali_prob[:, 1]
        ys = cali_labels_arr

        # Use Platt scaling to train a recalibrator.
        # calibrator = cal.PlattBinnerCalibrator(num_points, num_bins=num_bins)
        calibrator = cal.PlattCalibrator(num_points, num_bins=num_bins)
        calibrator.train_calibration(np.array(zs), ys)

        # Measure the calibration error of recalibrated model.
        calibrated_zs = calibrator.calibrate(test_pred_prob)
        ECE_after = cal.get_ece(calibrated_zs, test_labels_arr)
        print('before platt ece:{}%, after platt ece:{}%'.format(ECE_before*100, ECE_after*100))

        after_cali_test_prob = np.concatenate([1-calibrated_zs[...,np.newaxis], calibrated_zs[...,np.newaxis]], axis=1)
        if hparams.plot == True:
            fig, ax = draw_reliability_graph(after_cali_test_prob, test_labels, num_bins=num_bins, marginal=marginal)
            ax.title.set_text("Calibration: {}".format(hparams.calibrator))

    if hparams.plot == True:
        plt.show()

    return acc, ECE_before, ECE_after

if __name__ == "__main__":
    class Dotdict(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__

    import json
    # root_path = 'runs/cifar10/Ensemble/valid10000'
    root_path = 'runs/HO_predict/BNN/2hidden/v10_7BS'
    with open(root_path + '/hparams.json') as file:
        hparams_dict = json.load(file)
    hparams_dict["output_dir"] = root_path
    hparams_dict["sample_nbr"] = 10
    hparams_dict["calibrator"] = 'temperature_scaling'  # 'isotonic_regression' or 'temperature_scaling' or 'platt_scaling(for binary classification)'
    hparams_dict["num_bins"] = 15
    hparams_dict["marginal"] = False
    hparams_dict["cali_dataset_num"] = -1
    hparams_dict["plot"] = True
    hparams = Dotdict(hparams_dict)
    # main(hparams)

    acc_list, ECE_before_list, ECE_after_list = [], [], []
    for ii in range(1):
        _acc, _ECE_before, _ECE_after = main(hparams)
        acc_list.append(_acc)
        ECE_before_list.append(_ECE_before)
        ECE_after_list.append(_ECE_after)

    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    ECE_before_mean = np.mean(ECE_before_list)
    ECE_before_std = np.std(ECE_before_list)
    ECE_after_mean = np.mean(ECE_after_list)
    ECE_after_std = np.std(ECE_after_list)
    print('Acc: mean={}, std={}'.format(acc_mean, acc_std))
    print('ECE before cali: mean={}, std={}'.format(ECE_before_mean, ECE_before_std))
    print('ECE after cali: mean={}, std={}'.format(ECE_after_mean, ECE_after_std))