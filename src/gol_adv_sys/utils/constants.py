### TRAINING PARAMETERS #############################################################

# Batch size during training
bs = 64

# Number of training epochs
num_epochs = 100

# Number of new configurations to generate at each epoch
n_configs = 64*bs

# Max number of configurations in data set
n_max_configs = 15*n_configs

# Number of steps for training with a certain data set in an epoch
num_training_steps = 20

# Path to the folder where the results will be saved
results_folder_path = "results"

# Path to the folder where the models will be saved
models_folder_path = "models"


### SIMULATION PARAMETERS #########################################################

# Spatial size of training grids. All grixs will be resized to this size using a transformer.
grid_size = 32

# Number of cells in the generated grid
n_max_living_cells = grid_size * 2

# Number of steps to run the simulation
n_simulation_steps = 1

# Max number of steps to run the simulation
n_max_simulation_steps = 1000

TOPOLOGY = {
    "toroidal": "toroidal",
    "flat": "flat"
}



### MODELS PARAMETERS ##############################################################

# Number of channels in the training grids.
nc = 1

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 32

# Size of z latent vector (i.e. size of generator input)
nz = 256


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

