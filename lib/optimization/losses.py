import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Losses are from 
# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch


def DiceLoss(y_pred, y_real, smooth = 1):
    #comment out if your model contains a sigmoid or equivalent activation layer
    y_pred = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)
    
    #flatten label and prediction tensors
    y_pred = y_pred.view(-1)
    y_real = y_real.view(-1)
    
    intersection = (y_pred * y_real).sum()                            
    dice = (2.*intersection + smooth)/(y_pred.sum() + y_real.sum() + smooth) 
    
    return 1 - dice

def DiceBCELoss(y_pred, y_real, smooth=1):
    
    #comment out if your model contains a sigmoid or equivalent activation layer
    y_pred = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)
    
    #flatten label and prediction tensors
    y_pred = y_pred.reshape(-1)
    y_real = y_real.reshape(-1)
    
    intersection = (y_pred * y_real).sum()                            
    dice_loss = 1 - (2.*intersection + smooth)/(y_pred.sum() + y_real.sum() + smooth)  
    BCE = F.binary_cross_entropy(y_pred, y_real, reduction='mean')
    Dice_BCE = BCE + dice_loss
    
    return Dice_BCE


def IoULoss(y_pred, y_real, smooth=1):
        
    y_pred = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)      
    
    #flatten label and prediction tensors
    y_pred = y_pred.view(-1)
    y_real = y_real.view(-1)
    
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (y_pred * y_real).sum()
    total = (y_pred + y_real).sum()
    union = total - intersection 
    
    IoU = (intersection + smooth)/(union + smooth)
            
    return 1 - IoU

def loss2d_selector(config):
    """ Select the specified loss function
    """
    loss = config['loss_func']

    if loss == 'dice':
        return DiceLoss
    elif loss == 'diceBCE':
        return DiceBCELoss
    elif loss == 'IoU':
        return IoULoss
    else:
        raise Exception('Specified loss function not implemented!')
    return None