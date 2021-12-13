import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
import torchvision.transforms.functional as TF
import lib.optimization as opt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def visualize_test_equivariance(model, test_loader, config,
                      idx = 0, group = "C4"):
    """ Rotate images according to a group or at random and compute loss
    and visualize the resulting model output
    idx = is which image in the batch to show 
    group = "C4","C8","D4","D8" or None
    """
    # Set loss functions
    loss_fn = opt.loss2d_selector(config)
    
    model.eval()
    test_iter = iter(test_loader)
    X,y = next(test_iter)
    while idx > X.shape[0]-1:
        idx = idx-(X.shape[0])
        X,y = next(test_iter)
    # Set data to device
    X, y = X.to(device), y.to(device)

    if group == "C4" or group == "D4":
        rotations = [0, 90, 180, 270]
        figsize = (8, 7.3)
    elif group == "C8" or group == "D8":
        rotations = [ 45*x for x in np.arange(8) ]
        figsize = (14, 6.3)
    elif group is None:
        rotations = list(np.random.randint(0,360,5))
        rotations.sort()
        figsize = (12, 8.3)
    else:
        raise ValueError("Wrong group specification")
    
    f, ax = plt.subplots(3, len(rotations), figsize=figsize)
    # Rotate 4 times
    for i in range(len(rotations)):
        angle = int(rotations[i])
        
        # Rotate image and segmentation
        X_rot = TF.rotate(X, angle)
        y_rot = TF.rotate(y, angle)
        
        with torch.no_grad():
            y_pred = model(X_rot)
        loss = loss_fn(y_pred,y_rot)
    
        # squeeze output for visualization
        y_pred = torch.clamp(torch.sigmoid(y_pred), 1e-8, 1-1e-8)
        
        ax[0,i].imshow(np.moveaxis(X_rot[idx].cpu().numpy(),0,2))
        # ax[0,i].set_title(f"Rotated {angle}$^\circ$, loss: {round(loss.item(),4)}")
        ax[0,i].set_title(f"Rotated {angle}$^\circ$")
        # ax[0,i].set_xticklabels([])
        # ax[0,i].set_yticklabels([])
        ax[0,i].axis('off')
        ax[0,i].set_aspect('equal')
    
        ax[1,i].imshow(y_pred[idx, 0].cpu(), cmap='gray')
        ax[1,i].set_title(f'Loss: {round(loss.item(),3)}')
        # ax[1,i].axis('off')
        ax[1,i].set_xticklabels([])
        ax[1,i].set_yticklabels([])
        ax[1,i].set_aspect('equal')
        
        ax[2,i].imshow(y_rot[idx, 0].cpu(), cmap='gray')
        # ax[2,k].set_title('Real Segmentation')
        # ax[2,i].axis('off')
        ax[2,i].set_xticklabels([])
        ax[2,i].set_yticklabels([])
        ax[2,i].set_aspect('equal')
    ax[1,0].set_ylabel("Model")
    ax[2,0].set_ylabel("Ground Truth")
    plt.subplots_adjust(wspace=0, hspace=0)   
    return f,ax

def test_equivariance(model,test_loader, config,
                        group = "C4"):
    """ Iterates over a test loader and computes the
        loss and metrics for rotations according to a specified group
        or just random rotations
        
        group = "C4","C8","D4","D8" or None
    """
    
    # Set loss functions
    loss_fn = opt.loss2d_selector(config)
    
    if group == "C4" or group == "D4":
        rotations = [0, 90, 180, 270]
    elif group == "C8" or group == "D8":
        rotations = [ 45*x for x in np.arange(8) ]
    elif group is None:
        rotations = list(np.random.randint(0,360,5))
        rotations.sort()
    else:
        raise ValueError("Wrong group specification")
        
        
    # Initialize dict
    equivariance_dict = {}
    
    # Initialize losses
    test_loss = torch.zeros((len(rotations)))
    metrics_test = torch.zeros((len(rotations),5))
    
    for X,y in tqdm(test_loader,total=len(test_loader),
                    desc="Testing: ", position=0, leave=True):
        X, y = X.to(device), y.to(device)
        # Rotate dataset
        for k in range(len(rotations)):            
            angle = int(rotations[k])
        
            # Rotate image and segmentation
            X_rot = TF.rotate(X, angle)
            y_rot = TF.rotate(y, angle)
            
            # Model prediction
            with torch.no_grad():
                y_pred = model(X_rot)
            # Get losses
            loss = loss_fn(y_pred,y_rot).cpu()
            metrics = opt.compute_metrics(y_pred, y_rot)
            test_loss[k] += loss / len(test_loader)
            metrics_test[k] += metrics / len(test_loader)
    
    # Save metrics
    equivariance_dict['loss'] = test_loss
    equivariance_dict['metrics'] = metrics_test
    equivariance_dict['rotations'] = rotations
    
    return equivariance_dict