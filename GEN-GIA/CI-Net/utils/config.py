import torch

config = dict(
workers = 2, # Number of workers for dataloader
b_size = 128, # Batch size during training
patch = 0,
# total_img = 4, # Total image in client  
num_epochs = 5000, # Number of training epochs
lr = 0.001, # Learning rate for optimizers (Attacking)
tv_value = 0.0, # total variation value for inverting gradients loss
beta1 = 0.5,# Beta1 hyperparam for Adam optimizers
ngpu = 1, # Number of GPUs available. Use 0 for CPU mode.
device = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu"),
dst = "cifar10",# Configure the dst to use : "cifar10" or "imagenet"
lr_decay = False,    #decay_rate : Reduce the learning rate by decay rate
signed =True, # Updating gradients based on the sign only
saved_location = './Results',
classes = 100,

grad_idx = None,

LoRA = False,

CLIP = False,

# Defense

defense = "no_defense",
noise_std = 0.,
prun_p = 0.,
soteria_p = 0.,

fi = 40,
#----- Configuration for the generator in CI_Attacker -----------------------------#
nc = 3, # Number of channels in the training images. For color images this is 3
nz = 100, # Size of z latent vector (i.e. size of generator input)
ngf = 64,# Size of feature maps in generator
noise_setting = 'once', # noise setting  : 'once' is the generator will use same latent vector only to generate the reconstructed images    
rep_freq = 10, # report frequency: how frequent we shall save the data
)