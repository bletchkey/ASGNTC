### TRAINING PARAMETERS #############################################################

# Batch size during training
bs = 64

# Number of training epochs
num_epochs = 200

# Number of configurations to generate
n_configs = 4*bs

# Max number of configurations in data set
n_max_configs = 100*n_configs

# Number of steps for training
num_steps = 20


### SIMULATION PARAMETERS #########################################################

# Number of cells in the generated grid
n_max_living_cells = 16

# Number of steps to run the simulation
n_simulation_steps = 20

# Max number of steps to run the simulation
n_max_simulation_steps = 10000

# Spatial size of training grids. All grixs will be resized to this size using a transformer.
grid_size = 32


### MODELS PARAMETERS ##############################################################

# Number of channels in the training grids.
nc = 1

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 32

# Size of z latent vector (i.e. size of generator input)
nz = 256

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

