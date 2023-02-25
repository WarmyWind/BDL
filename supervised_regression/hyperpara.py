import argparse
from datetime import datetime
from lib.utils import get_results_directory, Hyperparameters, set_seed

parser = argparse.ArgumentParser()

########################### 通用参数 #############################
parser.add_argument(
    "--method",
    default="BNN",
    choices=["BNN", "MCDropout", "Ensemble", "EDL"],
    type=str,
    help="Specify method",
)

parser.add_argument("--noise_estimation",
                    type=str,
                    default='hetero',
                    choices=["none", "hetero"],
                    help="Type of noise estiamtion")

parser.add_argument(
    "--dataset_path",
    # default="Dataset/scene2_432UE_large_h_train_1004.npy",
    # default="Dataset/uci_energy_obs15_ftr15.npy",
    # default="Dataset/normed_8dB_50corr_large_fading_dB_dataset.npy",
    default="Dataset/normed_speed1_3_5ms_channel_dataset.npy",
    help="Dataset path",
)
parser.add_argument("--norm", type=bool, default=False, help="To norm dataset or not")

parser.add_argument("--task", default='ULA_channel',
                    choices=['large_channel_predict', 'ULA_channel', 'uci_wine', 'uci_crime', 'uci_energy'], type=str)
parser.add_argument("--data_gap", default=1, type=str, help="Gap of data (for abnormal detection test)")
parser.add_argument("--batch_size", type=int, default=2048, help="Batch size to use for training")
parser.add_argument("--epochs", type=int, default=2000, help="Training number of epochs")
parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--input_dim", type=int, default=5*32, help="input dimension")
parser.add_argument("--output_dim", type=int, default=1*32, help="output dimension")
parser.add_argument("--hidden_dim", type=int, default=512, help="NN hidden layer dimension")
parser.add_argument("--hidden_depth", type=int, default=3, help="Hidden layer depth for NN")
# parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")

timestamp = datetime.now().strftime("%Y-%m-%d-%A-%H-%M-%S")
parser.add_argument(
    "--output_dir",
    default='runs/' + parser.get_default("task") + '/' +parser.get_default("method") + '/' + timestamp,
    type=str,
    help="Specify output directory"
)

parser.add_argument("--seed", default=42, type=int, help="Seed to use for training")

########################### BNN\MCDropout参数 ##################################
parser.add_argument("--sample_nbr", type=int, default=5, help="Number of MC samples to get loss")

################################## BNN参数 ######################################
parser.add_argument("--only_output_layer", type=bool, default=True, help="If only output layer is bayesian")

########################### MCDropout参数 #############################
parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")

########################### Ensemble参数 ###############################
parser.add_argument("--num_net", type=int, default=10, help="Total number of nets of ensemble")


args = parser.parse_args()
hparams = Hyperparameters(**vars(args))
