### DATASET PARAMETERS ############################################################

# Number of configurations to generate for the dataset
DATASET_N_TOTAL_CONFIGS =  262400

# Ratio of the dataset to use for training
DATASET_TRAIN_RATIO = 0.8

# Ratio of the dataset to use for validation
DATASET_VAL_RATIO = 0.1

# Ratio of the dataset to use for testing
DATASET_TEST_RATIO = 0.1

# Batch size used in the generation of the dataset
DATASET_BATCH_SIZE = 128

# Target steps for the dataset
DATASET_N_SIM_STEPS = 1000

# The name of the dataset
DATASET_NAME = "gol_dataset"


### SIMULATION PARAMETERS ###############################################################

# Size of the grid
GRID_SIZE = 32

# Max number of steps to run the simulation
NUM_MAX_SIM_STEPS = 1000

# Number of living cells in the initial configuration
NUM_LIVING_CELLS_INITIAL = 16

# Threshold for the value of the cells in the generated configurations
THRESHOLD_CELL_VALUE = 0.5

# Half step for the targets
TARGET_EASY_HALF_STEP   = 2
TARGET_MEDIUM_HALF_STEP = 8
TARGET_HARD_HALF_STEP   = 100


### ADVERSARIAL TRAINING PARAMETERS #################################################

# Number of training epochs
NUM_ITERATIONS = 200

# Number of steps for training with a certain dataset in an epoch
NUM_TRAINING_STEPS = 4

# Batch size during training
ADV_BATCH_SIZE = 128

# Number of batches to generate at each epoch
NUM_BATCHES = 128

# Max number of batches in the dataloader
NUM_MAX_BATCHES = 16*NUM_BATCHES

# Number of new configurations to generate at each epoch
NUM_CONFIGS = NUM_BATCHES * ADV_BATCH_SIZE

# Max number of configurations in the dataloader
NUM_MAX_CONFIGS = NUM_MAX_BATCHES * ADV_BATCH_SIZE

# Average loss of P threshold before training G
THRESHOLD_AVG_LOSS_P = 0.05

# Number of configurations to generate for evaluation
NUM_CONFIGS_GEN_EVAL = 8192
NUM_BATCHES_GEN_EVAL = NUM_CONFIGS_GEN_EVAL // ADV_BATCH_SIZE

# Dirichlet alpha parameter - used for the Dirichlet distribution to generate the input for G
DIRICHLET_ALPHA = 0.3

# Number of steps to run the simulation for the generated configurations
NUM_SIM_STEPS = 1000


### PREDICTOR TRAINING PARAMETERS ###################################################

# Number of training epochs
P_NUM_EPOCHS = 100

# Batch size during training
P_BATCH_SIZE = 64

# Warmup phase parameters
WARMUP_TOTAL_STEPS = (int(DATASET_N_TOTAL_CONFIGS * DATASET_TRAIN_RATIO) // P_BATCH_SIZE)
WARMUP_INITIAL_LR  = 1e-6
WARMUP_TARGET_LR   = 0.01


### TRAINING GENERAL ##############################################################

# File name for monitoring the training progress
FILE_NAME_TRAINING_PROGRESS = "training_progress.txt"

# Key names for the checkpoint dictionary
CHECKPOINT_MODEL_STATE_DICT_KEY       = "state_dict"
CHECKPOINT_MODEL_OPTIMIZER_STATE_DICT = "optimizer_state_dict"
CHECKPOINT_MODEL_ARCHITECTURE_KEY     = "architecture"
CHECKPOINT_MODEL_TYPE_KEY             = "type"
CHECKPOINT_MODEL_NAME_KEY             = "name"
CHECKPOINT_EPOCH_KEY                  = "epoch"
CHECKPOINT_ITERATION_KEY              = "iter"
CHECKPOINT_TRAIN_LOSS_KEY             = "train_loss"
CHECKPOINT_VAL_LOSS_KEY               = "val_loss"
CHECKPOINT_SEED_KEY                   = "seed"
CHECKPOINT_DATE_KEY                   = "date"
CHECKPOINT_N_TIMES_TRAINED_KEY        = "n_times_trained"
CHECKPOINT_P_INPUT_TYPE               = "p_input_type"
CHECKPOINT_P_TARGET_TYPE              = "p_target_type"

# Hyperparameters for Adam optimizer
P_ADAM_LR  = 0.001
P_ADAM_B1  = 0.9
P_ADAM_B2  = 0.999
P_ADAM_EPS = 1e-08

G_ADAM_LR  = 0.001
G_ADAM_B1  = 0.9
G_ADAM_B2  = 0.999
G_ADAM_EPS = 1e-08

# Hyperparameters for AdamW optimizer
P_ADAMW_LR           = 0.001
P_ADAMW_B1           = 0.9
P_ADAMW_B2           = 0.999
P_ADAMW_EPS          = 1e-08
P_ADAMW_WEIGHT_DECAY = 0.01

G_ADAMW_LR           = 0.001
G_ADAMW_B1           = 0.9
G_ADAMW_B2           = 0.999
G_ADAMW_EPS          = 1e-08
G_ADAMW_WEIGHT_DECAY = 0.01

# Hyperparameters for SGD optimizer
P_SGD_LR           = 0.01
P_SGD_MOMENTUM     = 0.9
P_SGD_WEIGHT_DECAY = 1e-4


# NAMING CONVENTIONS #############################################################

# Types of topology for the grid
TOPOLOGY_TOROIDAL = "toroidal"
TOPOLOGY_FLAT     = "flat"

# Types of intialization for the grid
INIT_CONFIG_INTIAL_THRESHOLD = "threshold"
INIT_CONFIG_INITAL_N_CELLS   = "n_living_cells"
INIT_CONFIG_INITIAL_SIGN     = "sign"

# Training types
TRAINING_TYPE_PREDICTOR   = "training_predictor"
TRAINING_TYPE_ADVERSARIAL = "training_adversarial"

# Dataset keys
TRAIN      = "train"
VALIDATION = "validation"
TEST       = "test"

TRAIN_METADATA      = TRAIN      + "_metadata"
VALIDATION_METADATA = VALIDATION + "_metadata"
TEST_METADATA       = TEST       + "_metadata"

# Model keys
GENERATOR = "generator"
PREDICTOR = "predictor"

# Configuration keys
CONFIG_INITIAL       = "initial"
CONFIG_GENERATED     = "generated"
CONFIG_SIMULATED     = "simulated"
CONFIG_FINAL         = "final"
CONFIG_TARGET_EASY   = "easy"
CONFIG_TARGET_MEDIUM = "medium"
CONFIG_TARGET_HARD   = "hard"
CONFIG_TARGET_STABLE = "stable"

# Metadata keys
META_ID              = "id"
META_N_CELLS_INITIAL = "n_cells_init"
META_N_CELLS_FINAL   = "n_cells_final"
META_TRANSIENT_PHASE = "transient_phase"
META_PERIOD          = "period"

META_EASY_MIN = "easy_minimum"
META_EASY_MAX = "easy_maximum"
META_EASY_Q1  = "easy_q1"
META_EASY_Q2  = "easy_q2"
META_EASY_Q3  = "easy_q3"

META_MEDIUM_MIN = "medium_minimum"
META_MEDIUM_MAX = "medium_maximum"
META_MEDIUM_Q1  = "medium_q1"
META_MEDIUM_Q2  = "medium_q2"
META_MEDIUM_Q3  = "medium_q3"

META_HARD_MIN = "hard_minimum"
META_HARD_MAX = "hard_maximum"
META_HARD_Q1  = "hard_q1"
META_HARD_Q2  = "hard_q2"
META_HARD_Q3  = "hard_q3"

META_STABLE_MIN = "stable_minimum"
META_STABLE_MAX = "stable_maximum"
META_STABLE_Q1  = "stable_q1"
META_STABLE_Q2  = "stable_q2"
META_STABLE_Q3  = "stable_q3"


### MODELS PARAMETERS ###################################################################

# Number of channels in the grid (input and output)
NUM_CHANNELS_GRID = 1

# Size of feature maps in the generator
NUM_GENERATOR_FEATURES = GRID_SIZE

# Size of feature maps in the predictor
NUM_PREDICTOR_FEATURES = GRID_SIZE

# Size of z latent vector
LATENT_VEC_SIZE = 10

