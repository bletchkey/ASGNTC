### TRAINING PARAMETERS #################################################################

# Number of training epochs
NUM_EPOCHS = 200

# Batch size during training
BATCH_SIZE = 128

# Number of batches to generate at each epoch
N_BATCHES = 128

# Max number of batches in the dataset
N_MAX_BATCHES = 10*N_BATCHES

# Number of new configurations to generate at each epoch
N_CONFIGS = N_BATCHES * BATCH_SIZE

# Max number of configurations in dataset
N_MAX_CONFIGS = N_MAX_BATCHES * BATCH_SIZE

# Number of steps for training with a certain dataset in an epoch
NUM_TRAINING_STEPS = 20

# Average loss of P threshold before training G
THRESHOLD_AVG_LOSS_P = 0.05


### DATASET PARAMETERS ############################################################

# Number of configurations to generate for the dataset
DATASET_N_TOTAL_CONFIGS =  262400

# Ratio of the dataset to use for training
DATASET_TRAIN_RATIO = 0.8

# Ratio of the dataset to use for validation
DATASET_VAL_RATIO = 0.1

# Ratio of the dataset to use for testing
DATASET_TEST_RATIO = 0.1

# Batch size for the dataset
DATASET_BATCH_SIZE = 128

# Metric steps for the dataset
DATASET_N_SIM_STEPS = 1000

# The name of the dataset
DATASET_NAME = "gol_dataset"


# NAMING CONVENTIONS ####################################################################

# Configuration keys
CONFIG_INITIAL = "initial"
CONFIG_FINAL = "final"
CONFIG_SIMULATED = "simulated"
CONFIG_METRIC_EASY = "easy"
CONFIG_METRIC_MEDIUM = "medium"
CONFIG_METRIC_HARD = "hard"
CONFIG_METRIC_STABLE = "stable"

# Metadata keys
META_ID = "id"
META_N_CELLS_INIT = "n_cells_init"
META_N_CELLS_FINAL = "n_cells_final"
META_TRANSIENT_PHASE = "transient_phase"
META_PERIOD = "period"

META_EASY_MIN = "easy_minimum"
META_EASY_MAX = "easy_maximum"
META_EASY_Q1 = "easy_q1"
META_EASY_Q2 = "easy_q2"
META_EASY_Q3 = "easy_q3"

META_MEDIUM_MIN = "medium_minimum"
META_MEDIUM_MAX = "medium_maximum"
META_MEDIUM_Q1 = "medium_q1"
META_MEDIUM_Q2 = "medium_q2"
META_MEDIUM_Q3 = "medium_q3"

META_HARD_MIN = "hard_minimum"
META_HARD_MAX = "hard_maximum"
META_HARD_Q1 = "hard_q1"
META_HARD_Q2 = "hard_q2"
META_HARD_Q3 = "hard_q3"

META_STABLE_MIN = "stable_minimum"
META_STABLE_MAX = "stable_maximum"
META_STABLE_Q1 = "stable_q1"
META_STABLE_Q2 = "stable_q2"
META_STABLE_Q3 = "stable_q3"

# Types of topology for the grid
TOPOLOGY_TOROIDAL = "toroidal"
TOPOLOGY_FLAT = "flat"

# Types of intialization for the grid
INIT_CONFIG_INTIAL_THRESHOLD = "threshold"
INIT_CONFIG_INITAL_N_CELLS = "n_living_cells"


### SIMULATION PARAMETERS ###############################################################

# Size of the grid
GRID_SIZE = 32

# Number of steps to run the simulation
N_SIM_STEPS = 20

# Max number of steps to run the simulation
N_MAX_SIM_STEPS = 1000

# Number of living cells in the initial configuration
N_LIVING_CELLS_VALUE = 2 * GRID_SIZE

# Threshold for the value of the cells in the generated configurations
THRESHOLD_CELL_VALUE = 0.5

# Half step for the metrics
METRIC_EASY_HALF_STEP   = 2
METRIC_MEDIUM_HALF_STEP = 8
METRIC_HARD_HALF_STEP   = 100


### MODELS PARAMETERS ###################################################################

# Number of channels in the grid (input and output)
N_CHANNELS = 1

# Size of feature maps in the generator
N_GENERATOR_FEATURES = 32

# Size of feature maps in the predictor
N_PREDICTOR_FEATURES = 64

# Size of z latent vector
N_Z = 10

# Hyperparameters for Adam optimizer
P_ADAM_LR = 0.001
P_ADAM_B1 = 0.9
P_ADAM_B2 = 0.999
P_ADAM_EPS = 1e-08

G_ADAM_LR = 0.001
G_ADAM_B1 = 0.9
G_ADAM_B2 = 0.999
G_ADAM_EPS = 1e-08


# Hyperparameters for AdamW optimizer
P_ADAMW_LR = 0.001
P_ADAMW_B1 = 0.9
P_ADAMW_B2 = 0.999
P_ADAMW_EPS = 1e-08
P_ADAMW_WEIGHT_DECAY = 0.01

G_ADAMW_LR = 0.001
G_ADAMW_B1 = 0.9
G_ADAMW_B2 = 0.999
G_ADAMW_EPS = 1e-08
G_ADAMW_WEIGHT_DECAY = 0.01


# Hyperparameters for SGD optimizer
P_SGD_LR = 0.01
P_SGD_MOMENTUM = 0.9
P_SGD_WEIGHT_DECAY = 1e-4

# Warmup phase parameters
WARMUP_TOTAL_STEPS = 1000
WARMUP_INITIAL_LR = 1e-6
WARMUP_TARGET_LR = 0.1

