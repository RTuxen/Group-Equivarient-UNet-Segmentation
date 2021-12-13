import torch
import torch.nn as nn
from torchvision.transforms.functional import center_crop
from torchsummary import summary

def skip_connection(enc, dec):
    """ Combines encoder and decoder by cutting from the middle of
        the larger of the two and concatenate then
    """
    diff = enc.shape[-1]-dec.shape[-1]
    # Crop center
    if diff > 0:
        enc = center_crop(enc, output_size=dec.shape[2:])
    elif diff < 0:
        dec = center_crop(dec, output_size=enc.shape[2:])
    return torch.cat([enc, dec], 1)
    
class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
    
        # Mandatory input
        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.channels = config['channels']
        
        # Optional input
        self.n_conv = config['n_conv'] if 'n_conv' in config else 2
        self.kernel_size = config['kernel_size'] if 'kernel_size' in config else 3
        self.padding = config['padding'] if 'padding' in config else 1
        self.activation = config['activation'] if 'activation' in config else 'relu'
        self.dropout = config['dropout'] if 'dropout' in config else 0
        self.batch_norm = config['batch_norm'] if 'batch_norm' in config else False
    
        # Create encoder (downsampling)
        enc_dims = [self.in_channels, *self.channels]
        self.enc_conv = nn.ModuleList([])
        self.enc_pool = nn.ModuleList([])
        
        for i in range(1, len(enc_dims)):
            module_list = self.append_layer([], enc_dims[i-1], enc_dims[i],
                                            self.batch_norm,self.dropout,self.activation)
            for _ in range(self.n_conv-1):
                module_list = self.append_layer(module_list, enc_dims[i], enc_dims[i],
                                                self.batch_norm,self.dropout,self.activation)
            self.enc_conv.append(nn.Sequential(*module_list))
            
            # Create pooling seperatly
            self.enc_pool.append(
                 nn.Conv2d(enc_dims[i], enc_dims[i], kernel_size=2, stride=2, padding=0)
            )
                
        # Create bottleneck
        bottle_middle = []
        for _ in range(self.n_conv-1):
            bottle_middle = self.append_layer(bottle_middle, 2*enc_dims[-1], 2*enc_dims[-1],
                                                self.batch_norm,self.dropout,self.activation)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_dims[-1], 2*enc_dims[-1], kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU(),
            nn.Sequential(*bottle_middle),
            nn.Conv2d(2*enc_dims[-1], enc_dims[-1], kernel_size=self.kernel_size, padding=self.padding),
            nn.ReLU()
            )
    
        # decoder (upsampling)
        dec_dims = [self.out_channels, *self.channels]
        dec_dims.reverse()
        self.dec_conv = nn.ModuleList([])
        for i in range(1, len(dec_dims)):
            module_list = self.append_layer([], 2*dec_dims[i-1], dec_dims[i-1],
                                            self.batch_norm,self.dropout,self.activation)
            
            # Add convolution to bring channels down (except not for last layer)
            if dec_dims[i] == self.out_channels:
                module_list = self.append_layer(module_list, dec_dims[i-1], dec_dims[i-1],
                                                False,0,self.activation)
            else:
                module_list = self.append_layer(module_list, dec_dims[i-1], dec_dims[i],
                                                self.batch_norm,self.dropout,self.activation) 
                
            for _ in range(self.n_conv-2):
                # Add multiple convolutions in a level (except not for last layer)
                if dec_dims[i] == self.out_channels:
                    module_list = self.append_layer(module_list, dec_dims[i-1], dec_dims[i-1],
                                                    False,0,self.activation)
                else:
                    module_list = self.append_layer(module_list, dec_dims[i], dec_dims[i],
                                                    self.batch_norm,self.dropout,self.activation) 
                
            # final layer is without ReLU activation.
            if dec_dims[i] == self.out_channels:
                module_list = self.append_layer(module_list, dec_dims[i-1],dec_dims[i],
                                                False,0,activation=None)
            self.dec_conv.append(nn.Sequential(*module_list))
           
        # Create upsampling
        self.dec_upsample = nn.ModuleList([])
        for i in range(len(dec_dims)-1):
            self.dec_upsample.append(
                nn.ConvTranspose2d(dec_dims[i], dec_dims[i], kernel_size=4, stride=2, padding=1)
            )
        return None
    
    def append_layer(self, module_list, dim_1, dim_2, 
                     bn = False, dropout = 0, activation = 'relu'):
        """
        Creates a convolutional layer between two dimensions acording to 
        specifications in the class.
        """
        out_list = module_list
    
        # Add convolutional layer
        out_list.append(
            nn.Conv2d(dim_1, dim_2, 
                      kernel_size=self.kernel_size, padding=self.padding)
        )
    
        # add batch norm
        if bn:
            out_list.append(
                nn.BatchNorm2d(dim_2)
            )
    
        # add activation
        if activation is None:
            pass
        elif activation == 'relu':
            out_list.append(nn.ReLU())
        else:
            raise Exception('Specified activation function not implemented!')
    
        # add dropout
        if dropout != 0:
            out_list.append(nn.Dropout(p=dropout))
        
        return out_list

    def forward(self, x):
        enc = x # x is input of first encoder
        
        # Pass through the encoder
        enc_out = []
        for i in range(len(self.enc_conv)):
            # Pass through a single encoder layer
            enc = self.enc_conv[i](enc)

            # save the encoder output such that it can be used for skip connections
            enc_out.append(enc)

            # Downsample with convolutional pooling
            enc = self.enc_pool[i](enc)

        # Pass through the bottleneck
        b = self.bottleneck(enc)

        # Pass through the decoder
        dec = b
        enc_out.reverse()   # reverse such that it fits pass through decoder
        for i in range(len(self.dec_conv)):
            # Get input for decoder
            dec = self.dec_upsample[i](dec)
            dec = skip_connection(enc_out[i], dec)

            # Pass through a single decoder layer
            dec = self.dec_conv[i](dec)
        
        return dec

if __name__=='__main__': 
    if torch.cuda.is_available():
        print("The code will run on GPU.")
    else:
        print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'in_channels' : 3,              # Number of input channels
        'out_channels' : 1,             # Number of output channels
        'channels' : [4,8,16],        # Convolution channels in layers
        'n_conv' : 2,                   # Number of convolutions >= 2
        'batch_norm' : False,           # Use batch-norm (True/False)
        'dropout' : 0,                  # Use dropout? (0 <= dropout < 1)
        'kernel_size' : 3,              # Kernel size of all convolutions
        'padding' : 1                   # Padding on all convolutions
        }
    
    model = UNet(config).to(device)
    
    # x = torch.rand((4,3,128,128)).to(device)
    # y = model(x)
    summary(model,(3,128,128))
    
    
    
    