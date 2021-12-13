import torch
import numpy as np


def set_optimizers(model, config):
    """ Sets the optimizers for the training procedure
    """

    # Set optimizer
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    else: 
        raise Exception('Optimizer not implemented. Choose "adam" or "sgd".')
        
    if config['step_lr'][0]:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config['step_lr'][1], 
            gamma=config['step_lr'][2]
        )
    else:
        scheduler = None
        
    return optimizer, scheduler

#%% Early stopping

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    Code is from
    https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model = None
        self.early_stop = False
    def __call__(self, model, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = model
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.best_model = model
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


#%% Compute metrics
def compute_dice(TP, FP, FN): 
    # Compute dice score
    dice = 2 * TP / (2 * TP + FP + FN)
    return dice


def compute_iou(TP,FP,FN):
    """Computes the intersection over union of the predicted 
    and annotated segmentations"""  
    # https://en.wikipedia.org/wiki/Jaccard_index    
    iou = TP/(TP+FP+FN)

    return iou


def compute_accuracy(TP, TN, FP, FN):
    """Compute the accuracy: (TP + TN) / (TP + TN + FP + FN)
    Input
        TP  (int)   :   Number of true positives predicted
        TN  (int)   :   Number of true negatives predicted
        FP  (int)   :   Number of false positives predicted
        FN  (int)   :   Number of false negatives predicted
    """
    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc


def compute_sensitivity(TP, FN):
    """Compute the sensitivity: TP / (TP + FN)
    Inputs:
        TP  (int)   :   Number of true positives predicted
        FN  (int)   :   Number of false negatives predicted
    """
    # Compute sensitivity 
    sens = TP / (TP + FN)
    return sens


def compute_specificity(TN, FP):
    """Compute the specificity: TN / (TN + FP)
    Input
        TN  (int)   :   Number of true negatives predicted
        FP  (int)   :   Number of false positives predicted
        
    """
    #  Compute specificity
    spec = TN / (TN + FP)
    return spec

def compute_metrics(y_pred, y_real):
    """
    Computes metrics for semantic segmentation

    Parameters
    ----------
    y_pred : tensor
        the output from a segmentation model. Should not have any 
        activation (sigmoid,softmax etc.) used.
    y_real : tensor
        the ground truth segmentation values should be [0,1].

    Returns
    -------
    numpy array
        array with the following:
            [DICE-score, IoU-Score, Accuracy, Sensitivity, Specificity]

    """
    # Perform activation and move to cpu
    y_pred = torch.clamp(torch.sigmoid(y_pred).detach().cpu(), 1e-8, 1-1e-8) 
    y_real = y_real.detach().cpu()
        
    #flatten label and prediction tensors
    y_pred = y_pred.view(-1)
    y_real = y_real.view(-1)
    
    #True Positives, False Positives, False Negatives & True Negatives
    TP = (y_pred * y_real).sum()
    FP = ((1-y_real) * y_pred).sum()
    FN = (y_real * (1-y_pred)).sum()
    TN = ((1-y_real) * (1-y_pred)).sum()
    
    # Compute performance metrics
    dice = compute_dice(TP, FP, FN).numpy()
    iou = compute_iou(TP,FP,FN).numpy()
    acc = compute_accuracy(TP, TN, FP, FN).numpy()
    sens = compute_sensitivity(TP, FN).numpy()
    spec = compute_specificity(TN, FP).numpy()
    return np.array([dice, iou, acc, sens, spec])
