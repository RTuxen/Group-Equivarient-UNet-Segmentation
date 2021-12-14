import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output
from tqdm import tqdm
import numpy as np
import time
import lib.optimization as opt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def warm_up(model):
    """ Warm up a model by making a forward pass
    """
    wrmup = model(torch.randn(1, 3, 128, 128).to(device))
    del wrmup
    return None

def train(model, config, train_loader, val_loader,
            plotting=True):
    """ Trains 2D UNet model,
        Should be provided with a config file with the following elements:
            'early_stopping' : 5 (int)
            'epochs' : 10 (int)
            'learning_rate': 1e-2,
            'optimizer': 'adam',
            'step_lr': [True, 5, 0.1],
            'loss_func' : 'diceBCE'/'IoU'/'dice'
    """
    
    # Set loss functions
    loss_fn = opt.loss2d_selector(config)
    
    # Set optimizer and scheduler
    optimizer, scheduler = opt.set_optimizers(model,config)
    
    # Set early stopping
    if config['early_stopping'] is not None:
        early_stopping = opt.EarlyStopping(patience=config['early_stopping'])
    else:
        early_stopping = None
    
    if plotting : clear_output(wait=True)
    
    # Warm up model
    warm_up(model)
    
    # Initialize dicts
    train_dict = {'epochs' : 0,
                   'lr' : [scheduler.get_last_lr()[0]],
                  'loss': [],
                  'metrics': []}
    val_dict = {'loss': [],
                'metrics': []}
    
    
    # Start timer
    start = time.time()
    
    for epoch in range(config['epochs']):
        # break
        print(f"* Epoch {epoch+1}/{config['epochs']}")
        
        model.train()
        
        # Training pass
        train_loss = 0
        metrics_train = torch.tensor([0, 0, 0, 0, 0])
        for X_train, y_train in tqdm(train_loader, desc='Training: ',
                                     position=0,leave=True):
            # break
            X_train, y_train = X_train.to(device), y_train.to(device)
        
            # set parameter gradients to zero
            optimizer.zero_grad()
        
            # model pass
            y_pred = model(X_train)
            
            # update
            loss = loss_fn(y_pred, y_train) # forward-pass
            loss.backward()                 # backward-pass
            optimizer.step()                # update weights
        
            # calculate metrics to show the user
            train_loss += loss.item() / len(train_loader)
            metrics_train += opt.compute_metrics(y_pred, y_train) / len(train_loader)
    
        # Step the learning rate
        if scheduler is not None:
            scheduler.step()
            
        # Compute validation loss
        model.eval()
        val_loss = 0
        metrics_val = torch.tensor([0, 0, 0, 0, 0])
        for X_val, y_val in tqdm(val_loader, desc='Validating : ',
                                 position=0,leave=True):
            X_val, y_val = X_val.to(device), y_val.to(device)
            with torch.no_grad():
                y_pred = model(X_val)
    
            loss = loss_fn(y_pred, y_val).cpu().item()
            
            # Compute loss and metrics
            val_loss += loss / len(val_loader)
            metrics_val += opt.compute_metrics(y_pred, y_val) / len(val_loader)
            
            
        # Plot annotations against model predictions on validation data
        if plotting:
            # Get some validation data
            X_val, y_val = next(iter(val_loader))
            with torch.no_grad():
                y_hat = torch.sigmoid(model(X_val.to(device))).detach().cpu()
            
            # Show 6 or less images
            N_imgs = y_hat.shape[0] if y_hat.shape[0] < 6 else 6
            imgsize = N_imgs*3 if N_imgs*3 < 14 else 14
            
            # Plot
            clear_output(wait=True)
            f, ax = plt.subplots(3, N_imgs, figsize=(imgsize, 6))
            ax = ax.flatten()
            for k in range(N_imgs):                
                ax[k].imshow(np.moveaxis(X_val[k].numpy(),0,2))
                ax[k].set_title('Real data')
                ax[k].axis('off')
                ax[k+N_imgs].imshow(y_hat[k, 0], cmap='gray')
                ax[k+N_imgs].set_title('Model Output')
                ax[k+N_imgs].axis('off')
                ax[k+2*N_imgs].imshow(y_val[k, 0], cmap='gray')
                ax[k+2*N_imgs].set_title('Real Segmentation')
                ax[k+2*N_imgs].axis('off')
            plt.suptitle('%d / %d - loss: %f' % (epoch+1, config['epochs'], val_loss))
            plt.show()
            
        # save loss in dicts
        train_dict['loss'].append(train_loss)
        val_dict['loss'].append(val_loss)  
        # Add metrics to dicts
        train_dict['metrics'].append(metrics_train.numpy())
        val_dict['metrics'].append(metrics_val.numpy())
        
        # Save lr
        train_dict['lr'].append(scheduler.get_last_lr()[0])
        
        # Perform early stopping
        if early_stopping is not None:
            early_stopping(model, val_loss)
            if early_stopping.early_stop:
                model = early_stopping.best_model
                break
        
    # Take the model with the best performance on the validation set
    if early_stopping is not None:
        model = early_stopping.best_model
        
    # Stop timer and save runtime
    end = time.time()
    runtime = end - start
    train_dict['runtime'] = runtime
    # Save number of epochs
    train_dict['epochs'] = epoch+1
    
    # Convert to numpy arrays
    train_dict['loss'] = np.array(train_dict['loss'])
    val_dict['loss'] = np.array(val_dict['loss'])
    val_dict['metrics'] = np.array(val_dict['metrics'])
    train_dict['metrics'] = np.array(train_dict['metrics'])
    train_dict['lr'] = np.array(train_dict['lr'])
    return train_dict, val_dict

def test(model, test_loader, config):
    """ Test the model on a testing set
    """
    
    # Set loss functions
    loss_fn = opt.loss2d_selector(config)
    
    # Initialize dict
    test_dict = {}
    
    model.eval()
    test_loss = 0
    metrics_test = torch.tensor([0, 0, 0, 0, 0])
    for X_test, y_test in tqdm(test_loader, desc='Testing : ',
                               position=0,leave=True):
        X_test, y_test = X_test.to(device), y_test.to(device)
        with torch.no_grad():
            y_pred = model(X_test)

        loss = loss_fn(y_pred, y_test).cpu().item()
        
        # Compute loss and metrics
        test_loss += loss / len(test_loader)
        metrics_test += opt.compute_metrics(y_pred, y_test) / len(test_loader)
    
    # Save metrics
    test_dict['loss'] = test_loss
    test_dict['metrics'] = np.array(metrics_test.numpy())
    
    return test_dict