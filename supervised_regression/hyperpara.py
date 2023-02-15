import argparse
from datetime import datetime
from lib.utils import get_results_directory, Hyperparameters, set_seed

parser = argparse.ArgumentParser()

########################### 通用参数 #############################
parser.add_argument(
    "--method",
    default="MCDropout",
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
    default="Dataset/uci_energy_obs15_ftr15.npy",
    help="Dataset path",
)

parser.add_argument("--task", default='uci_energy', type=str, help="Specify task")
parser.add_argument("--data_gap", default=1, type=str, help="Gap of data (for abnormal detection test)")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use for training")
parser.add_argument("--epochs", type=int, default=500, help="Training number of epochs")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--input_dim", type=int, default=5*3, help="input dimension")
parser.add_argument("--output_dim", type=int, default=1, help="output dimension")
parser.add_argument("--hidden_dim", type=int, default=256, help="NN hidden layer dimension")
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

########################### MCDropout参数 #############################
parser.add_argument("--dropout_rate", type=float, default=0.5, help="Dropout rate")

########################### Ensemble参数 ###############################
parser.add_argument("--num_net", type=int, default=10, help="Total number of nets of ensemble")


args = parser.parse_args()
hparams = Hyperparameters(**vars(args))
