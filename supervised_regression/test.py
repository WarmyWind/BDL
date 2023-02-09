import torch
from torch.utils.tensorboard.writer import SummaryWriter
# import torch_geometric
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
from bnn.model import BayesianRegressor
from mcdropout.model import MCDropoutRegressor, log_gaussian_loss
from ensemble.model import BootstrapEnsemble
from edl.model import EvidentialRegressor
from lib.plot_func import *
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import uncertainty_toolbox as uct
from Dataset.load_dataset import *
# Set plot style
uct.viz.set_style()
uct.viz.update_rc("text.usetex", True)  # Set to True for system latex
uct.viz.update_rc("font.size", 14)  # Set font size
uct.viz.update_rc("xtick.labelsize", 14)  # Set font size for xaxis tick labels
uct.viz.update_rc("ytick.labelsize", 14)  # Set font size for yaxis tick labels

def main(hparams):
    if hparams.task == 'large_channel_predict':
        dl_train, dl_calibration, dl_test = load_large_channel(hparams.dataset_path, hparams.input_dim, hparams.output_dim,
                                                         test_size=0.2, valid_size=0.2,
                                                         train_batch_size=hparams.batch_size,
                                                         valid_batch_size=hparams.batch_size, seed=hparams.seed)

    elif hparams.task == 'uci_wine':
        dl_train, dl_calibration, dl_test = load_uci_wine(test_size=0.25, valid_size=0,
                                                    train_batch_size=hparams.batch_size,
                                                    valid_batch_size=hparams.batch_size, seed=hparams.seed)

    elif hparams.task == 'uci_crime':
        dl_train, dl_calibration, dl_test = load_uci_crime(test_size=0.25, valid_size=0,
                                                     train_batch_size=hparams.batch_size,
                                                     valid_batch_size=hparams.batch_size, seed=hparams.seed)

    else:
        raise Exception('Invalid task!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method = hparams.method  # 'BNN' , 'MCDropout'

    lr = hparams.learning_rate
    # epochs = hparams.epochs
    # results_dir = hparams.output_dir
    # writer = SummaryWriter(log_dir=str(results_dir))

    input_dim = hparams.input_dim
    hidden_dim = hparams.hidden_dim
    output_dim = hparams.output_dim
    dropout_rate = hparams.dropout_rate
    if method =='BNN':
        if hparams.noise_estimation == 'none':
            model = BayesianRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      hidden_depth=hparams.hidden_depth, hetero_noise_est=False)
        elif hparams.noise_estimation == 'hetero':
            model = BayesianRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      hidden_depth=hparams.hidden_depth, hetero_noise_est=True)
        else:
            raise Exception("Unsupported noise estimation: \"{}\"!".format(hparams.noise_estimation))
        if model.hetero_noise_est == False:
            criterion = torch.nn.MSELoss()
        else:
            criterion = log_gaussian_loss

    elif method == 'MCDropout':
        if hparams.noise_estimation == 'none':
            model = MCDropoutRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                       pdrop=dropout_rate, hidden_depth=hparams.hidden_depth, hetero_noise_est=False)
        elif hparams.noise_estimation == 'hetero':
            model = MCDropoutRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                       pdrop=dropout_rate, hidden_depth=hparams.hidden_depth, hetero_noise_est=True)
        else:
            raise Exception("Unsupported noise estimation: \"{}\"!".format(hparams.noise_estimation))

    elif method == 'Ensemble':
        num_net = hparams.num_net
        if hparams.noise_estimation == 'none':
            model = BootstrapEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      hidden_depth=hparams.hidden_depth, learn_rate=lr, weight_decay=1e-4,
                                      num_net=num_net, hetero_noise_est=False)
        elif hparams.noise_estimation == 'hetero':
            model = BootstrapEnsemble(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                      hidden_depth=hparams.hidden_depth, learn_rate=lr, weight_decay=1e-4,
                                      num_net=num_net, hetero_noise_est=True)
        else:
            raise Exception("Unsupported noise estimation: \"{}\"!".format(hparams.noise_estimation))

    elif method == 'EDL':
        model = EvidentialRegressor(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,
                                  hidden_depth=hparams.hidden_depth)
    else:
        raise Exception('Invalid method!')

    if method != "Ensemble":
        state_dict = torch.load(hparams.output_dir+'/model.pt')
        model.load_state_dict(state_dict)
        model.to(device)
    else:
        model.load(hparams.output_dir)
        model.to(device)

    def test_step(batch):
        if method != 'MCDropout':
            model.eval()

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        if method == 'MCDropout':
            y_mean, y_std, loss, mse = model.sample_detailed_loss(inputs=x,
                                                 labels=y,
                                                 sample_nbr=hparams.sample_nbr)

            mse, mape = model.get_accuracy_matrix(inputs=x,
                                    labels=y,
                                    sample_nbr=hparams.sample_nbr)

        elif method == 'BNN':
            complexity_cost_weight = 1 / len(dl_train)
            y_mean, y_std, loss, mse = model.sample_detailed_loss(inputs=x,
                                                 labels=y,
                                                 criterion=criterion,
                                                 sample_nbr=hparams.sample_nbr,
                                                 complexity_cost_weight=complexity_cost_weight)

            mse, mape = model.get_accuracy_matrix(inputs=x,
                                                  labels=y,
                                                  sample_nbr=hparams.sample_nbr)

        elif method == 'Ensemble':
            y_mean, y_std, loss, mse = model.sample_detailed_loss(inputs=x, labels=y)
            mse, mape = model.get_accuracy_matrix(inputs=x, labels=y)

        elif method == 'EDL':
            y_mean, y_std, loss, mse = model.sample_detailed_loss(inputs=x, labels=y)
            mse, mape = model.get_accuracy_matrix(inputs=x, labels=y)
        else:
            raise Exception('Invalid method!')

        rmse = mse ** 0.5
        return y_mean, y_std, rmse, mape

    for batch in dl_test:
        x = batch[0].detach().cpu().numpy()
        y = batch[1].detach().cpu().numpy()
        y_mean, y_std, rmse, mape = test_step(batch)
        print("RMSE:{}, MAPE:{}".format(rmse, mape))
        break

    y_mean = y_mean.squeeze()
    y_std = y_std.squeeze()
    y = y.squeeze()

    # Before recalibration
    prop_type = 'interval'  # ‘interval’ or 'quantile'
    exp_props, obs_props = uct.get_proportion_lists_vectorized(
        y_mean, y_std, y, prop_type=prop_type
    )
    mace = uct.mean_absolute_calibration_error(
        y_mean, y_std, y, recal_model=None, prop_type=prop_type
    )
    rmsce = uct.root_mean_squared_calibration_error(
        y_mean, y_std, y, recal_model=None, prop_type=prop_type
    )
    ma = uct.miscalibration_area(
        y_mean, y_std, y, recal_model=None, prop_type=prop_type
    )

    print("Before Recalibration:  ", end="")
    MSCE_before = rmsce ** 2
    print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

    if hparams.plot == True:
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        uct.plot_calibration(
            y_mean,
            y_std,
            y,
            exp_props=exp_props,
            obs_props=obs_props,
            ax=ax1,
        )
        ax1.title.set_text("Before Calibration")

        # plot a sample before calibration
        fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(10, 5))
        test_idx = 250  # 200, 250, 300
        x_test = x[test_idx]
        test_confidence = 0.3
        ax2_1 = plot_sample_uncertainty(x_test, y_mean, y_std, y, test_idx, in_exp_proportions=test_confidence, ax=ax2_1)
        ax2_1.title.set_text("Before Calibration")
        ax2_1.set_ylabel("Value")
        # plt.show()

    ############################# Calibration #############################
    for batch in dl_calibration:
        y_cali = batch[1].detach().cpu().numpy()
        y_cali_mean, y_cali_std, _, _ = test_step(batch)
        break

    y_cali_mean = y_cali_mean.squeeze()
    y_cali_std = y_cali_std.squeeze()
    y_cali = y_cali.squeeze()

    # Before recalibration
    exp_props, obs_props = uct.get_proportion_lists_vectorized(
        y_cali_mean, y_cali_std, y_cali, prop_type=prop_type
    )
    recal_model = uct.iso_recal(exp_props, obs_props)
    recal_exp_props, recal_obs_props = uct.get_proportion_lists_vectorized(
        y_mean, y_std, y, recal_model=recal_model, prop_type=prop_type
    )
    mace = uct.mean_absolute_calibration_error(
        y_mean, y_std, y, recal_model=recal_model, prop_type=prop_type
    )
    rmsce = uct.root_mean_squared_calibration_error(
        y_mean, y_std, y, recal_model=recal_model, prop_type=prop_type
    )
    ma = uct.miscalibration_area(
        y_mean, y_std, y, recal_model=recal_model, prop_type=prop_type
    )

    print("After Recalibration:  ", end="")
    MSCE_after = rmsce ** 2
    print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

    if hparams.plot == True:
        # plot a sample after calibration
        x_test = x[test_idx]
        ax2_2 = plot_sample_uncertainty(x_test, y_mean, y_std, y, test_idx, in_exp_proportions=test_confidence, ax=ax2_2, recal_model=recal_model)
        ax2_2.title.set_text("After Calibration")

        # plot confidence curve after calibration
        uct.plot_calibration(
            y_mean,
            y_std,
            y,
            exp_props=recal_exp_props,
            obs_props=recal_obs_props,
            ax=ax2,
        )
        ax2.title.set_text("After Calibration")
        plt.show()
        # plt.close()

    return rmse, mape, MSCE_before, MSCE_after

if __name__ == "__main__":
    class Dotdict(dict):
        __setattr__ = dict.__setitem__
        __getattr__ = dict.__getitem__

    import json
    root_path = 'runs/large_channel_predict/EDL/hetero_noise_est'
    with open(root_path + '/hparams.json') as file:
        hparams_dict = json.load(file)
    hparams_dict["output_dir"] = root_path
    hparams_dict["sample_nbr"] = 1
    hparams_dict["plot"] = True
    hparams = Dotdict(hparams_dict)
    # main(hparams)
    test_times = 1

    mape_list, rmse_list, MSCE_before_list, MSCE_after_list = [], [], [], []
    for ii in range(test_times):
        _rmse, _mape, _MSCE_before, _MSCE_after = main(hparams)
        rmse_list.append(_rmse)
        mape_list.append(_mape)
        MSCE_before_list.append(_MSCE_before)
        MSCE_after_list.append(_MSCE_after)

    rmse_mean = np.mean(rmse_list)
    rmse_std = np.std(rmse_list)
    mape_mean = np.mean(mape_list)
    mape_std = np.std(mape_list)
    MSCE_before_mean = np.mean(MSCE_before_list)
    MSCE_before_std = np.std(MSCE_before_list)
    MSCE_after_mean = np.mean(MSCE_after_list)
    MSCE_after_std = np.std(MSCE_after_list)
    print('\nTest Result:')
    print('RMSE: mean={}, std={}'.format(rmse_mean, rmse_std))
    print('MAPE: mean={}, std={}'.format(mape_mean, mape_std))
    print('MSCE before cali: mean={}, std={}'.format(MSCE_before_mean, MSCE_before_std))
    print('MSCE after cali: mean={}, std={}'.format(MSCE_after_mean, MSCE_after_std))