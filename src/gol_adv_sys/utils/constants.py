### TRAINING PARAMETERS #############################################################

# Number of training epochs
num_epochs = 200

# Batch size during training
bs = 128

# Number of batches to generate at each epoch
n_batches = 128

# Max number of batches in the dataset
n_max_batches = 10*n_batches

# Number of new configurations to generate at each epoch
n_configs = n_batches*bs

# Max number of configurations in dataset
n_max_configs = n_max_batches*bs

# Number of steps for training with a certain dataset in an epoch
num_training_steps = 20

# Average loss of P threshold before training G
threshold_avg_loss_p = 0.005

# Base path to the folder where the trainings will be saved
trainings_folder_path = "trainings"

# Path to the folder where the results will be saved
results_folder_path = "results"

# Path to the folder where the models will be saved
models_folder_path = "models"

# Path to the folder where the logs will be saved
logs_folder_path = "logs"

# Path to the folder where the trained predictor is saved
trained_models_path = "trained_models"


### FIXED DATASET PARAMETERS ##############################################################

# Number of configurations to generate for the fixed dataset
fixed_dataset_n_configs =  262400

# Ratio of the fixed dataset to use for training
fixed_dataset_train_ratio = 0.8

# Ratio of the fixed dataset to use for validation
fixed_dataset_val_ratio = 0.1

# Ratio of the fixed dataset to use for testing
fixed_dataset_test_ratio = 0.1

# Path to the folder where the fixed dataset is saved
fixed_dataset_path = "data"

# Batch size for the fixed dataset
fixed_dataset_bs = 128

# Metric steps for the fixed dataset
fixed_dataset_n_simulation_steps = 1000

# Fixed dataset name
fixed_dataset_name = "gol_fixed_dataset"

### SIMULATION PARAMETERS #########################################################

# Size of the grid
grid_size = 32

# Number of steps to run the simulation
n_simulation_steps = 10

# Max number of steps to run the simulation
n_max_simulation_steps = 1000

# Number of living cells in the initial configuration
n_living_cells = 2 * grid_size

# Threshold for the value of the cells in the generated configurations
threshold_cell_value = 0.5

# Names for each configuration
CONFIG_NAMES = {
    "initial": "initial",
    "final": "final",
    "metric_easy": "easy",
    "metric_medium": "medium",
    "metric_hard": "hard"
}

# Type of topology for the grid
TOPOLOGY_TYPE = {
    "toroidal": "toroidal",
    "flat": "flat"
}

# Type of intialization for the grid
INIT_CONFIG_TYPE = {
    "n_living_cells": "n_living_cells",
    "threshold": "threshold"
}

# Type of metric for the grid
METRIC_TYPE = {
    "easy": CONFIG_NAMES["metric_easy"],
    "medium": CONFIG_NAMES["metric_medium"],
    "hard": CONFIG_NAMES["metric_hard"]
}


### MODELS PARAMETERS ##############################################################

# Number of channels in the grid (input and output)
nc = 1

# Size of feature maps in the generator
ngf = 32

# Size of feature maps in the predictor
npf = 64

# Size of z latent vector (i.e. size of generator input)
nz = 10

# Hyperparameters for Adam optimizer
p_adam_lr=0.001
p_adam_b1=0.9
p_adam_b2=0.999
p_adam_eps=1e-08

g_adam_lr=0.001
g_adam_b1=0.9
g_adam_b2=0.999
g_adam_eps=1e-08

# Hyperparameters for AdamW optimizer
p_adamw_lr=0.001
p_adamw_b1=0.9
p_adamw_b2=0.999
p_adamw_eps=1e-08
p_adamw_wd=0.01

g_adamw_lr=0.001
g_adamw_b1=0.9
g_adamw_b2=0.999
g_adamw_eps=1e-08
g_adamw_wd=0.01

# Hyperparameters for SGD optimizer
p_sgd_lr=0.01
p_sgd_momentum=0.9
p_sgd_wd=1e-4

