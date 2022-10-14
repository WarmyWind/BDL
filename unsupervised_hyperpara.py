import argparse
from datetime import datetime
from lib.utils import get_results_directory, Hyperparameters, set_seed

parser = argparse.ArgumentParser()

########################### 通用参数 #############################
parser.add_argument(
    "--method",
    default="GNN",
    choices=["GNN", "GNN+GP"],
    type=str,
    help="Specify method",
)
parser.add_argument("--task", default='sum_rate_maximization', type=str, help="Specify task")
# parser.add_argument(
#     "--dataset_path",
#     default="Dataset/scene2_432UE_large_h_train_1004.npy",
#     help="Dataset path",
# )

parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use for training")
parser.add_argument("--epochs", type=int, default=300, help="Training number of epochs")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--input_dim", type=int, default=15, help="input dimension")
parser.add_argument("--output_dim", type=int, default=15, help="output dimension")
parser.add_argument("--hidden_dim", type=int, default=256, help="NN hidden layer dimension")
parser.add_argument("--hidden_depth", type=int, default=1, help="Hidden layer depth for NN")
# parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")

timestamp = datetime.now().strftime("%Y-%m-%d-%A-%H-%M-%S")
parser.add_argument(
    "--output_dir",
    default='runs/' + parser.get_default("task") + '/' +parser.get_default("method") + '/' + timestamp,
    type=str,
    help="Specify output directory"
)


# ########################### BNN\MCDropout参数 ##################################
# parser.add_argument("--sample_nbr", type=int, default=5, help="Number of MC samples to get loss")

########################### MCDropout\DUE\SNGP参数 #############################
parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")

########################### GP参数 #######################################
parser.add_argument(
    "--n_inducing_points", type=int, default=20, help="Number of inducing points"
)

# parser.add_argument(
#     "--n_power_iterations",
#     type=int,
#     default=1,
#     help="Number of power iterations for Resnet",
# )

# parser.add_argument(
#     "--coeff",
#     type=float,
#     default=0.95,
#     help="Coefficient for Resnet",
# )

parser.add_argument(
    "--kernel",
    default="RBF",
    choices=["RBF", "RQ", "Matern12", "Matern32", "Matern52"],
    help="Pick a kernel",
)

# parser.add_argument(
#     "--spectral_normalization",
#     action="store_true",
#     help="Use spectral normalization",
# )

# parser.add_argument(
#     "--no_spectral_conv",
#     action="store_false",
#     dest="spectral_conv",
#     help="Don't use spectral normalization on the convolutions",
# )

# parser.add_argument(
#     "--no_spectral_bn",
#     action="store_false",
#     dest="spectral_bn",
#     help="Don't use spectral normalization on the batch normalization layers",
# )

# parser.add_argument("--seed", type=int, help="Seed to use for training")

args = parser.parse_args()
hparams = Hyperparameters(**vars(args))
