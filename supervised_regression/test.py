import torch
# import torch_geometric
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
from utils import build_model, get_dataloader, get_abnormal_dataloader
# Set plot style
uct.viz.set_style()
uct.viz.update_rc("text.usetex", True)  # Set to True for system latex
uct.viz.update_rc("font.size", 14)  # Set font size
uct.viz.update_rc("xtick.labelsize", 14)  # Set font size for xaxis tick labels
uct.viz.update_rc("ytick.labelsize", 14)  # Set font size for yaxis tick labels

def main(hparams):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl_train, dl_calibration, dl_test = get_dataloader(hparams)

    hparams.data_gap = 3
    noisy_feature = False
    hparams.dataset_path = "Dataset/uci_energy_obs25_ftr25.npy"
    _, _, dl_abnormal = get_abnormal_dataloader(hparams, noisy_feature)

    method = hparams.method  # 'BNN' , 'MCDropout'
    model = build_model(hparams)

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

    # Test on Abnormal Set
    test_labels = torch.tensor([])
    test_y_mean, test_y_std = np.empty((0, hparams.output_dim)), np.empty((0, hparams.output_dim))
    rmse, mape = 0, 0
    for batch in dl_abnormal:
        # x = batch[0].detach().cpu().numpy()
        # _label = batch[1].detach().cpu().numpy()
        test_labels = torch.cat((test_labels, batch[1]))
        _y_mean, _y_std, _rmse, _mape = test_step(batch)
        test_y_mean = np.concatenate((test_y_mean, _y_mean), 0)
        test_y_std = np.concatenate((test_y_std, _y_std), 0)
        rmse += _rmse**2 * len(batch[1])
        mape += _mape * len(batch[1])
    rmse = (rmse/len(test_labels))**0.5
    mape = mape/len(test_labels)
    mse_arr = np.mean(np.square(test_y_mean - test_labels.detach().cpu().numpy()), axis=0)
    rmse_arr = np.sqrt(mse_arr)
    std_arr = np.mean(test_y_std, axis=0)
    print("Abnormal test: RMSE={}\n      std={}\n".format(rmse_arr, std_arr))

    # Test on Test Set
    test_labels = torch.tensor([])
    test_y_mean, test_y_std = np.empty((0, hparams.output_dim)), np.empty((0, hparams.output_dim))
    rmse, mape = 0, 0
    for batch in dl_test:
        # x = batch[0].detach().cpu().numpy()
        # _label = batch[1].detach().cpu().numpy()
        test_labels = torch.cat((test_labels, batch[1]))
        _y_mean, _y_std, _rmse, _mape = test_step(batch)
        test_y_mean = np.concatenate((test_y_mean, _y_mean), 0)
        test_y_std = np.concatenate((test_y_std, _y_std), 0)
        rmse += _rmse**2 * len(batch[1])
        mape += _mape * len(batch[1])
    rmse = (rmse/len(test_labels))**0.5
    mape = mape/len(test_labels)
    mse_arr = np.mean(np.square(test_y_mean - test_labels.detach().cpu().numpy()), axis=0)
    rmse_arr = np.sqrt(mse_arr)
    std_arr = np.mean(test_y_std, axis=0)
    print("Test: RMSE={}\n      std={}\n".format(rmse_arr, std_arr))

    data_raw_shape = test_y_mean.shape
    test_y_mean = test_y_mean.flatten().squeeze()
    test_y_std = test_y_std.flatten().squeeze()
    test_labels = test_labels.flatten().squeeze().detach().cpu().numpy()

    # Before recalibration
    prop_type = 'interval'  # ‘interval’ or 'quantile'
    exp_props, obs_props = uct.get_proportion_lists_vectorized(
        test_y_mean, test_y_std, test_labels, prop_type=prop_type
    )
    mace = uct.mean_absolute_calibration_error(
        test_y_mean, test_y_std, test_labels, recal_model=None, prop_type=prop_type
    )
    rmsce = uct.root_mean_squared_calibration_error(
        test_y_mean, test_y_std, test_labels, recal_model=None, prop_type=prop_type
    )
    ma = uct.miscalibration_area(
        test_y_mean, test_y_std, test_labels, recal_model=None, prop_type=prop_type
    )

    print("Before Recalibration:  ", end="")
    MSCE_before = rmsce ** 2
    print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

    if hparams.plot == True:
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        uct.plot_calibration(
            test_y_mean,
            test_y_std,
            test_labels,
            exp_props=exp_props,
            obs_props=obs_props,
            ax=ax1,
        )
        ax1.title.set_text("Before Calibration")

        # plot a sample before calibration
        fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(10, 5))
        test_idx = 10
        x_test = dl_test.dataset.__getitem__(test_idx)[0].detach().cpu().numpy()
        if hparams.task == 'uci_energy':
            x_test = x_test[:, -1]
        test_confidence = 0.3
        ax2_1 = plot_sample_uncertainty(x_test, test_y_mean.reshape(data_raw_shape), test_y_std.reshape(data_raw_shape),
                                        test_labels.reshape(data_raw_shape), test_idx,
                                        in_exp_proportions=test_confidence, ax=ax2_1)
        ax2_1.title.set_text("Before Calibration")
        ax2_1.set_ylabel("Value")

    ############################# Calibration #############################
    cali_labels = torch.tensor([])
    cali_y_mean, cali_y_std = np.empty((0, hparams.output_dim)), np.empty((0, hparams.output_dim))
    for batch in dl_calibration:
        # _cali_labels = batch[1].detach().cpu().numpy()
        cali_labels = torch.cat((cali_labels, batch[1]))
        _y_mean, _y_std, _, _ = test_step(batch)
        cali_y_mean = np.concatenate((cali_y_mean, _y_mean), 0)
        cali_y_std = np.concatenate((cali_y_std, _y_std), 0)

    cali_y_mean = cali_y_mean.squeeze().flatten()
    cali_y_std = cali_y_std.squeeze().flatten()
    cali_labels = cali_labels.squeeze().flatten().detach().cpu().numpy()

    # Recalibration
    exp_props, obs_props = uct.get_proportion_lists_vectorized(
        cali_y_mean, cali_y_std, cali_labels, prop_type=prop_type
    )
    recal_model = uct.iso_recal(exp_props, obs_props)

    # Test calibration performance after recalibration
    recal_exp_props, recal_obs_props = uct.get_proportion_lists_vectorized(
        test_y_mean, test_y_std, test_labels, recal_model=recal_model, prop_type=prop_type
    )
    mace = uct.mean_absolute_calibration_error(
        test_y_mean, test_y_std, test_labels, recal_model=recal_model, prop_type=prop_type
    )
    rmsce = uct.root_mean_squared_calibration_error(
        test_y_mean, test_y_std, test_labels, recal_model=recal_model, prop_type=prop_type
    )
    ma = uct.miscalibration_area(
        test_y_mean, test_y_std, test_labels, recal_model=recal_model, prop_type=prop_type
    )

    print("After Recalibration:  ", end="")
    MSCE_after = rmsce ** 2
    print("MACE: {:.5f}, RMSCE: {:.5f}, MA: {:.5f}".format(mace, rmsce, ma))

    if hparams.plot == True:
        # plot a sample after calibration
        # x_test = x[test_idx]
        ax2_2 = plot_sample_uncertainty(x_test, test_y_mean.reshape(data_raw_shape), test_y_std.reshape(data_raw_shape),
                                        test_labels.reshape(data_raw_shape), test_idx, in_exp_proportions=test_confidence,
                                        ax=ax2_2, recal_model=recal_model)
        ax2_2.title.set_text("After Calibration")

        # plot confidence curve after calibration
        uct.plot_calibration(
            test_y_mean, test_y_std, test_labels,
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
    root_path = 'runs/uci_energy/MCDropout/input05_output=1'
    with open(root_path + '/hparams.json') as file:
        hparams_dict = json.load(file)
    hparams_dict["output_dir"] = root_path
    hparams_dict["sample_nbr"] = 10
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