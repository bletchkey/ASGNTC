# Batch size during training
bs = 64

# Number of training epochs
num_epochs = 20

# Number of steps in each epoch
nun_steps = 10

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 32

# Number of channels in the training images.
nc = 1

# Size of feature maps in generator
ngf = 16

# Size of feature maps in discriminator
ndf = 16

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Number of cells in the generated grid
n_max_living_cells = image_size

# Number of steps to run the simulation
n_simulation_steps = 50

# Hyperparameters for Adam optimizer
p_adam_lr = 0.0001 # 0.0002
p_adam_b1 = 0.5 # 0.9
p_adam_b2 = 0.999
p_adam_eps = 1e-08

g_adam_lr = 0.0001
g_adam_b1 = 0.5
g_adam_b2 = 0.999
g_adam_eps = 1e-08


#Hyperparameters for SGD optimizer
p_sgp_lr = 0.01
p_sgp_m = 0.9

g_sgp_lr = 0.01
g_sgp_m = 0.9


#Hyperparameters for AdamW optimizer
p_adamw_lr=0.001
p_adamw_b1=0.9
p_adamw_b2=0.999
p_adamw_eps=1e-08
p_adamw_wd=0.004

g_adamw_lr=0.0001
g_adamw_b1=0.5
g_adamw_b2=0.999
g_adamw_eps=1e-08
g_adamw_wd=0.004


