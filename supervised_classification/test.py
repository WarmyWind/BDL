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
# from lib.evaluate_uncertainty import loss_divided_by_MCoutput, loss_divided_by_uncertainty
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
    if hparams.task == 'HO_predict':
        Data_set = np.load(hparams.dataset_path, allow_pickle=True).tolist()
        X = Data_set['x']
        y = Data_set['y']
        X = X[:, -hparams.input_dim:]
        y = y[:, :hparams.output_dim]
        X = StandardScaler().fit_transform(X)
        y = y.flatten()
        # y = StandardScaler().fit_transform(y)

        # 80% data for training process, 20% for model calibration
        X, X_calibration, y, y_calibration = train_test_split(X, y, test_size=.2, random_state=hparams.seed)

        # In the 80%, 70% for training/evaluating, 10% for testing
        X, X_test, y, y_test = train_test_split(X, y, test_size=.1 / 0.8, random_state=hparams.seed)

        # In the 70%, 60% for training, 10% for validation
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.1 / 0.7, random_state=hparams.seed)

        X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
        X_valid, y_valid = torch.tensor(X_valid).float(), torch.tensor(y_valid).float()
        X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()
        X_calibration, y_calibration = torch.tensor(X_calibration).float(), torch.tensor(y_calibration).float()

        ds_train = torch.utils.data.TensorDataset(X_train, y_train)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=len(ds_train), shuffle=True)
        ds_valid = torch.utils.data.TensorDataset(X_valid, y_valid)
        dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=len(ds_valid), shuffle=True)
        ds_test = torch.utils.data.TensorDataset(X_test, y_test)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=len(ds_test))
        ds_calibration = torch.utils.data.TensorDataset(X_calibration, y_calibration)
        dl_calibration = torch.utils.data.DataLoader(ds_calibration, batch_size=len(ds_calibration))
    else:
        raise Exception('Invalid task!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    method = hparams.method  # 'BNN' , 'MCDropout'

    lr = hparams.learning_rate
    epochs = hparams.epochs
    results_dir = hparams.output_dir
    # writer = SummaryWriter(log_dir=str(results_dir))

    input_dim = hparams.input_dim
    hidden_dim = hparams.hidden_dim
    output_dim = hparams.output_dim
    dropout_rate = hparams.dropout_rate
    if method == 'BNN':
        model = BayesianClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                  hidden_depth=hparams.hidden_depth)
        criterion = F.cross_entropy

    elif method == 'MCDropout':
        model = MCDropoutClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                   pdrop=dropout_rate, hidden_depth=hparams.hidden_depth)

        # loss_fn = F.mse_loss
    elif method == 'Ensemble':
        num_net = hparams.num_net

        model = BootstrapEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                  hidden_depth=hparams.hidden_depth, learn_rate=lr, weight_decay=1e-4,
                                  num_net=num_net, task='classify')

    elif method == 'EDL':
        model = EvidentialClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                     hidden_depth=hparams.hidden_depth, annealing_step=hparams.annealing_step)
        if hparams.edl_loss == 'digamma':
            criterion = edl_digamma_loss
        elif hparams.edl_loss == 'log':
            criterion = edl_log_loss
        elif hparams.edl_loss == 'mse':
            criterion = edl_mse_loss

    elif method == 'DNN':
        try :
            binary_classifier = hparams.binary_classifier
            if not binary_classifier:
                loss_func = F.cross_entropy
                model = DNNClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      hidden_depth=hparams.hidden_depth, learn_rate=lr, weight_decay=1e-4,
                                      loss_func=loss_func)
            else:
                loss_func = nn.BCEWithLogitsLoss()
                model = DNNBinaryClassifier(input_dim=input_dim, hidden_dim=hidden_dim, hidden_depth=hparams.hidden_depth,
                                            learn_rate=lr, weight_decay=1e-4, loss_func=loss_func)
        except:
            loss_func = F.cross_entropy
            model = DNNClassifier(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                  hidden_depth=hparams.hidden_depth, learn_rate=lr, weight_decay=1e-4,
                                  loss_func=loss_func)
    else:
        raise Exception('Invalid method!')

    if method != "Ensemble":
        state_dict = torch.load(hparams.output_dir+'/model.pt')
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
            complexity_cost_weight = 1 / X_train.shape[0]
            y_pred, y_prob, _, acc = model.sample_detailed_loss(inputs=x,
                                                      labels=y,
                                                      criterion=criterion,
                                                      sample_nbr=hparams.sample_nbr,
                                                      complexity_cost_weight=complexity_cost_weight)

        elif method == 'Ensemble':
            y_pred, y_prob, _, acc = model.sample_detailed_loss(inputs=x, labels=y)

        elif method == 'EDL':
            y_pred, y_prob, _, acc = model.sample_detailed_loss(inputs=x, labels=y, criterion=criterion, epoch=hparams.epochs)

        elif method == 'DNN':
            y_pred, y_prob, acc = model.sample_detailed_loss(inputs=x, labels=y)

        else:
            raise Exception('Invalid method!')

        return y_pred, y_prob, acc

    ################################### Before Calibration ########################################
    for batch in dl_test:
        # x = batch[0].detach().cpu().numpy()
        test_labels = batch[1]
        y_pred, y_prob, acc = test_step(batch, model)
        print("Acc: {}".format(acc))
        break

    y_pred = y_pred.squeeze()
    y_prob = y_prob.squeeze()

    # Before recalibration, test
    num_bins = hparams.num_bins
    marginal = hparams.marginal
    test_labels_arr = test_labels.detach().cpu().numpy().astype('int32')
    if hparams.calibrator == 'platt_scaling':
        test_pred_prob = y_prob[:, 1]
        ECE_before = cal.get_ece(test_pred_prob, test_labels_arr)
    fig, ax = draw_reliability_graph(y_prob, test_labels, num_bins=num_bins, marginal=marginal)
    ECE_before, MCE_before = get_metrics(y_prob, test_labels, num_bins, marginal)
    ax.title.set_text("Before Calibration")
    plt.show()

    ############################# Calibration #############################
    if hparams.calibrator == 'isotonic_regression':
        for batch in dl_calibration:
            # x_cali = batch[0].detach().cpu().numpy()
            cali_labels = batch[1]
            y_cali_pred, y_cali_prob, _ = test_step(batch, model)
            break

        y_cali_pred = y_cali_pred.squeeze()
        y_cali_prob = y_cali_prob.squeeze()
        bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(y_cali_prob, cali_labels, num_bins, marginal)
        bin_accs = np.concatenate([[0], bin_accs, [1]])
        bin_confs = np.concatenate([[0], bin_confs, [1]])
        recal_model = uct.iso_recal(bin_accs, bin_confs)
        fig, ax = draw_reliability_graph(y_prob, test_labels, num_bins=num_bins, recal_model=recal_model, marginal=marginal)

        _shape = y_prob.shape
        probs = recal_model.predict(y_prob.flatten())
        probs = np.reshape(probs, _shape)
        ECE_after, MCE_after = get_metrics(probs, test_labels, num_bins, marginal)
        ax.title.set_text("Calibration:{}".format(hparams.calibrator))

    elif hparams.calibrator == 'temperature_scaling':
        scaled_model = ModelWithTemperature(model)
        scaled_model.set_temperature(dl_calibration, sample_nbr=hparams.sample_nbr)

        for x, labels in dl_test:
            x, labels = x.to(device), labels.to(device)
            y_pred, y_prob, _acc = scaled_model.sample_detailed_loss(x, labels, hparams.sample_nbr)
            break

        fig, ax = draw_reliability_graph(y_prob, test_labels, num_bins=num_bins, marginal=marginal)
        ECE_after, MCE_after = get_metrics(y_prob, test_labels, num_bins, marginal)
        ax.title.set_text("Calibration: {}".format(hparams.calibrator))

    elif hparams.calibrator == 'platt_scaling':
        for batch in dl_calibration:
            # x_cali = batch[0].detach().cpu().numpy()
            cali_labels = batch[1]
            y_cali_pred, y_cali_prob, _ = test_step(batch, model)
            break

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
        fig, ax = draw_reliability_graph(after_cali_test_prob, test_labels, num_bins=num_bins, marginal=marginal)
        ax.title.set_text("Calibration: {}".format(hparams.calibrator))
    plt.show()
    return acc, ECE_before, ECE_after

if __name__ == "__main__":
    class Dotdict(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__

    import json
    root_path = 'runs/HO_predict/BNN/2023-01-18-Wednesday-23-02-58'
    # root_path = 'runs/HO_predict/BNN/2hidden/v1+v5+v10_7BS'
    with open(root_path + '/hparams.json') as file:
        hparams_dict = json.load(file)
    hparams_dict["output_dir"] = root_path
    hparams_dict["sample_nbr"] = 10
    hparams_dict["calibrator"] = 'isotonic_regression'  # 'isotonic_regression' or 'temperature_scaling' or 'platt_scaling(for binary classification)'
    hparams_dict["num_bins"] = 15
    hparams_dict["marginal"] = False
    hparams = Dotdict(hparams_dict)
    main(hparams)

    # acc_list, ECE_before_list, ECE_after_list = [], [], []
    # for ii in range(100):
    #     _acc, _ECE_before, _ECE_after = main(hparams)
    #     acc_list.append(_acc)
    #     ECE_before_list.append(_ECE_before)
    #     ECE_after_list.append(_ECE_after)
    #
    # acc_mean = np.mean(acc_list)
    # acc_std = np.std(acc_list)
    # ECE_before_mean = np.mean(ECE_before_list)
    # ECE_before_std = np.std(ECE_before_list)
    # ECE_after_mean = np.mean(ECE_after_list)
    # ECE_after_std = np.std(ECE_after_list)
    # print('Acc: mean={}, std={}'.format(acc_mean, acc_std))
    # print('ECE before cali: mean={}, std={}'.format(ECE_before_mean, ECE_before_std))
    # print('ECE after cali: mean={}, std={}'.format(ECE_after_mean, ECE_after_std))