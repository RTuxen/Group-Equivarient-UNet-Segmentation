import os
import sys
sys.path.append('../')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # Only needed for Spyder
import torch
import lib.models as m



if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
""" The purpose of this script is to compare the number of trainable
    parameters in the UNet and G-UNet with different groups.
    These parameters are used for a fair comparison of the networks
"""


#%%

config1 = {
    # G-Unet parameters
    'in_channels' : 3,              # Number of input channels
    'out_channels' : 1,             # Number of output channels
    'channels' : [6,10,12],         # Convolution channels in layers
    'n_conv' : 2,                   # Number of convolutions >= 2
    'batch_norm' : True,            # Use batch-norm (True/False)
    'dropout' : 0,                  # Use dropout? (0 <= dropout < 1)
    'equvariant_mask' : False,      # Whether to mask for equivariance
    'group'   : "C4"                # Rotation group
}
config2 = {
    # G-Unet parameters
    'in_channels' : 3,              # Number of input channels
    'out_channels' : 1,             # Number of output channels
    'channels' : [4,6,9],           # Convolution channels in layers
    'n_conv' : 2,                   # Number of convolutions >= 2
    'batch_norm' : True,            # Use batch-norm (True/False)
    'dropout' : 0,                  # Use dropout? (0 <= dropout < 1)
    'equvariant_mask' : False,      # Whether to mask for equivariance
    'group'   : "C8"                # Rotation group
}
config3 = {
    # G-Unet parameters
    'in_channels' : 3,              # Number of input channels
    'out_channels' : 1,             # Number of output channels
    'channels' : [4,6,9],           # Convolution channels in layers
    'n_conv' : 2,                   # Number of convolutions >= 2
    'batch_norm' : True,            # Use batch-norm (True/False)
    'dropout' : 0,                  # Use dropout? (0 <= dropout < 1)
    'equvariant_mask' : False,      # Whether to mask for equivariance
    'group'   : "D4"                # Rotation group
}
config4 = {
    # G-Unet parameters
    'in_channels' : 3,              # Number of input channels
    'out_channels' : 1,             # Number of output channels
    'channels' : [3,5,6],           # Convolution channels in layers
    'n_conv' : 2,                   # Number of convolutions >= 2
    'batch_norm' : True,            # Use batch-norm (True/False)
    'dropout' : 0,                  # Use dropout? (0 <= dropout < 1)
    'equvariant_mask' : False,      # Whether to mask for equivariance
    'group'   : "D8"                # Rotation group
}
config5 = {
    # Unet parameters
    'in_channels' : 3,              # Number of input channels
    'out_channels' : 1,             # Number of output channels
    'channels' : [6,12,20],         # Convolution channels in layers
    'n_conv' : 2,                   # Number of convolutions >= 2
    'batch_norm' : True,            # Use batch-norm (True/False)
    'dropout' : 0                   # Use dropout? (0 <= dropout < 1)
}
#%% Get models

models = [m.G_UNet(config1), m.G_UNet(config2), m.G_UNet(config3),
          m.G_UNet(config4), m.UNet(config5)]

for model in models:
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("N params:",pytorch_total_params)