import argparse
from datetime import datetime
from lib.utils import get_results_directory, Hyperparameters, set_seed

parser = argparse.ArgumentParser()

########################### 通用参数 #############################
parser.add_argument(
    "--method",
    default="Ensemble",
    choices=["BNN", "MCDropout", "Ensemble", "EDL", 'DNN', 'CNN'],
    type=str,
    help="Specify method",
)
parser.add_argument(
    "--task",
    default='cifar10',
    choices=["HO_predict", "cifar10"],
    type=str,
    help="Specify task"
)

parser.add_argument(
    "--dataset_path",
    default="Dataset/v1+v5+v10_7BS_HO_supervised_60000data.npy",
    help="Dataset path",
)
# parser.add_argument("--noise_estimation",
#                     type=str,
#                     default='none',
#                     choices=["none", "hetero"],
#                     help="Type of noise estiamtion")

parser.add_argument("--binary_classifier", type=bool, default=False, help="use binary classifier")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size to use for training")
parser.add_argument("--epochs", type=int, default=200, help="Training number of epochs")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--input_dim", type=int, default=(32,32), help="input dimension")
parser.add_argument("--output_dim", type=int, default=10, help="output dimension, i.e. the number of classification")
parser.add_argument("--hidden_dim", type=int, default=120, help="NN hidden layer dimension")
parser.add_argument("--hidden_depth", type=int, default=5, help="Hidden layer depth for NN")
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
parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")

########################### Ensemble参数 ###############################
parser.add_argument("--num_net", type=int, default=10, help="Total number of nets of ensemble")
parser.add_argument("--dataset_ratio", type=float, default=0.5, help="Dataset ratio for bagging")

########################### EDL参数 ###############################
parser.add_argument("--edl_loss", type=str, default='digamma', help="Loss type of EDL, digamma or log or mse")
parser.add_argument("--annealing_step", type=int, default=200, help="Annealing step of EDL")

args = parser.parse_args()
hparams = Hyperparameters(**vars(args))
